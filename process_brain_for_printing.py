#!/usr/bin/env python3

"""
This script is designed to generate 3D printable brain surfaces (pial + brainstem) 
in both T1 and MNI space. It also provides an option to color these surfaces using 
a parameter map (e.g., a segmentation or functional data). 

By default, the script is quiet (i.e., it hides all console outputs from the 
external software it calls). If you provide '--verbose', it will print the 
commands and their output.

Usage example:
    process_brain_for_printing.py --subjects_dir /path/to/derivatives \
                                  --subject_id sub-01 \
                                  --param_map /path/to/some_volume.nii.gz \
                                  --export_obj \
                                  --use_midthickness
"""

import argparse
import subprocess
import glob
import os
import shutil
import uuid

import nibabel as nib
import numpy as np

from scipy.ndimage import map_coordinates
from skimage import measure
import trimesh
import matplotlib

# ===========================================================================
# 1) HELPER FUNCTIONS
# ===========================================================================
def run_cmd(cmd, verbose=False):
    """
    Run an external command as a subprocess.

    If 'verbose' is True, the command and its output are shown.
    Otherwise, all external messages are suppressed.
    """
    if verbose:
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def first_match(pattern):
    """
    Return the first file that matches the given pattern.
    If multiple matches are found, return the first but warn the user.
    """
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[WARNING] Multiple files found for {pattern}; using {matches[0]}")
    return matches[0]


# ===========================================================================
# 2) CREATE MRTRIX-STYLE WARP (warp.nii)
# ===========================================================================
def generate_mrtrix_style_warp(mni_file, t1_file, xfm_file, out_warp="warp.nii", tmp_dir=".", verbose=False):
    """
    Generate a 3D warp field that transforms coordinates from MNI to T1 
    in the format expected by MRtrix tools. The three components of the 
    warp (x, y, z) are stacked into a 4D NIfTI file.
    """
    print("[INFO] Generating MRtrix-style warp field ...")

    no_warp_0 = os.path.join(tmp_dir, "no-warp-0.nii")
    no_warp_1 = os.path.join(tmp_dir, "no-warp-1.nii")
    no_warp_2 = os.path.join(tmp_dir, "no-warp-2.nii")
    warp_0    = os.path.join(tmp_dir, "warp_0.nii")
    warp_1    = os.path.join(tmp_dir, "warp_1.nii")
    warp_2    = os.path.join(tmp_dir, "warp_2.nii")

    # Initialize empty warp images
    run_cmd(["warpinit", mni_file, os.path.join(tmp_dir, "no-warp-[].nii")], verbose=verbose)

    # Apply transforms for each dimension
    for i in range(3):
        run_cmd([
            "antsApplyTransforms", "-d", "3",
            "-i", f"{tmp_dir}/no-warp-{i}.nii",
            "-r", t1_file,
            "-o", f"{tmp_dir}/warp_{i}.nii",
            "-t", xfm_file,
            "-n", "Linear"
        ], verbose=verbose)

    # Concatenate into a single 4D volume
    run_cmd([
        "mrcat",
        warp_0,
        warp_1,
        warp_2,
        os.path.join(tmp_dir, out_warp)
    ], verbose=verbose)

    print(f"[INFO] Created MRtrix-style warp => {os.path.join(tmp_dir, out_warp)}")


# ===========================================================================
# 3) WARPING A GIFTI SURFACE
# ===========================================================================
def warp_gifti_vertices(gifti_file, warp_field_file, output_gifti_file, verbose=False):
    """
    Apply a warp field (4D NIfTI, shape [X, Y, Z, 3]) to each vertex of a GIFTI surface.
    The warp is in MNI->T1 orientation, so the result is a surface in the T1 space.
    """
    print(f"[INFO] Warping GIFTI surface: {os.path.basename(gifti_file)}")

    # Load the GIFTI
    gifti_img = nib.load(gifti_file)
    vertices_darray = gifti_img.darrays[0]
    vertices = vertices_darray.data  # shape (N, 3)

    # Load the warp field
    warp_img = nib.load(warp_field_file)
    warp_data = warp_img.get_fdata()  # shape (X, Y, Z, 3)
    warp_affine = warp_img.affine
    inv_affine = np.linalg.inv(warp_affine)

    # Convert vertices to voxel coords of warp volume
    ones = np.ones((vertices.shape[0], 1))
    vert_hom = np.hstack([vertices, ones])  # shape (N, 4)
    vox_hom  = (inv_affine @ vert_hom.T).T
    vox_xyz  = vox_hom[:, :3]

    # For each dimension, sample the warp volume
    warped_coords = np.zeros_like(vertices)
    for dim in range(3):
        vol_dim = warp_data[..., dim]
        sample_coords = [vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]]
        warped_coords[:, dim] = map_coordinates(
            vol_dim,
            sample_coords,
            order=1,  # tri-linear sampling
            mode='nearest'
        )

    # Save new vertex coords
    vertices_darray.data = warped_coords.astype(np.float32)
    nib.save(gifti_img, output_gifti_file)
    print(f"[INFO] Warped GIFTI => {os.path.basename(output_gifti_file)}")


# ===========================================================================
# 4) GIFTI -> TRIMESH
# ===========================================================================
def gifti_to_trimesh(gifti_file):
    """
    Read a GIFTI surface file and return it as a trimesh.Trimesh object.
    """
    gii = nib.load(gifti_file)
    verts = gii.darrays[0].data
    faces = gii.darrays[1].data
    return trimesh.Trimesh(vertices=verts, faces=faces)


# ===========================================================================
# 5) MARCHING CUBES => GIFTI
# ===========================================================================
def volume_to_gifti(nifti_file, out_gifti, level=0.5):
    """
    Convert a binary mask (NIfTI) into a GIFTI surface mesh using marching_cubes.
    The 'level' argument is the isosurface value (usually 0.5 for a binarized mask).
    """
    print(f"[INFO] Running marching_cubes on: {os.path.basename(nifti_file)}")

    # Load the volume and do marching cubes
    nii = nib.load(nifti_file)
    vol = nii.get_fdata()
    affine = nii.affine

    verts_vox, faces, _, _ = measure.marching_cubes(vol, level=level)

    # Convert voxel coords -> world (scanner) coords
    ones = np.ones((verts_vox.shape[0], 1))
    vert_vox_hom = np.hstack([verts_vox, ones])
    vert_xyz_hom = affine @ vert_vox_hom.T
    vert_xyz = vert_xyz_hom[:3, :].T

    # Build GIFTI from (x, y, z) + faces
    gii = nib.gifti.GiftiImage()
    coords_da = nib.gifti.GiftiDataArray(
        data=vert_xyz.astype(np.float32),
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT32'
    )
    faces_da = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_INT32'
    )
    gii.darrays.extend([coords_da, faces_da])
    nib.save(gii, out_gifti)
    print(f"[INFO] Saved GIFTI => {os.path.basename(out_gifti)}")


# ===========================================================================
# 6) EXTRACT BRAINSTEM
# ===========================================================================
def extract_brainstem_in_t1(subjects_dir, subject_id, tmp_dir=".", verbose=False):
    """
    Locate the freesurfer aseg file in T1 space, binarize certain labels 
    to isolate 'brainstem' region, and convert to GIFTI using marching_cubes.
    """
    anat_dir = os.path.join(subjects_dir, subject_id, "anat")
    aseg_nii_pattern = f"{anat_dir}/*_run-01_desc-aseg_dseg.nii.gz"
    aseg_nii = first_match(aseg_nii_pattern)

    # Convert to a temporary T1 mask
    t1_brainstem_tmp  = os.path.join(tmp_dir, "brainstem_tmp_T1.nii.gz")
    run_cmd(["mri_convert", aseg_nii, t1_brainstem_tmp], verbose=verbose)

    # Binarize the relevant labels
    final_mask_nii = os.path.join(tmp_dir, "brainstem_bin_T1.nii.gz")
    BRAINSTEM_LABELS = [2,3,24,31,41,42,63,72,77,51,52,13,12,43,50,4,11,26,58,49,10,17,18,53,54,44,5,80,14,15,30,62]
    match_str = [str(lbl) for lbl in BRAINSTEM_LABELS]

    run_cmd([
        "mri_binarize",
        "--i", t1_brainstem_tmp,
        "--match"
    ] + match_str + [
        "--inv",
        "--o", os.path.join(tmp_dir, "bin_tmp_T1.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", t1_brainstem_tmp, "-mul",
        os.path.join(tmp_dir, "bin_tmp_T1.nii.gz"),
        os.path.join(tmp_dir, "brainstem_mask_T1.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", os.path.join(tmp_dir, "brainstem_mask_T1.nii.gz"),
        "-bin", final_mask_nii
    ], verbose=verbose)

    # Convert the mask to a surface
    out_gii = os.path.join(tmp_dir, "brainstem_in_t1.surf.gii")
    volume_to_gifti(final_mask_nii, out_gii, level=0.5)
    return out_gii


def extract_brainstem_in_mni(subjects_dir, subject_id, out_aseg_in_mni, tmp_dir=".", verbose=False):
    """
    Similar to 'extract_brainstem_in_t1' but for MNI space. 
    We warp the aseg from T1 to MNI, binarize the brainstem labels, 
    and convert to GIFTI with marching_cubes.
    """
    anat_dir = os.path.join(subjects_dir, subject_id, "anat")
    aseg_nii_pattern = f"{anat_dir}/*_run-01_desc-aseg_dseg.nii.gz"
    xfm_file_pattern = f"{anat_dir}/*_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    mni_template_pattern = f"{anat_dir}/*_run-01_space-MNI152NLin2009cAsym_*_T1w.nii.gz"

    aseg_nii     = first_match(aseg_nii_pattern)
    xfm_file     = first_match(xfm_file_pattern)
    mni_template = first_match(mni_template_pattern)

    # Warp aseg -> MNI
    run_cmd([
        "antsApplyTransforms", "-d", "3",
        "-i", aseg_nii,
        "-o", out_aseg_in_mni,
        "-r", mni_template,
        "-t", xfm_file,
        "-n", "NearestNeighbor"
    ], verbose=verbose)

    final_mask_nii = os.path.join(tmp_dir, "brainstem_bin_mni.nii.gz")
    tmp_nii        = os.path.join(tmp_dir, "brainstem_tmp_mni.nii.gz")
    run_cmd(["mri_convert", out_aseg_in_mni, tmp_nii], verbose=verbose)

    BRAINSTEM_LABELS = [2,3,24,31,41,42,63,72,77,51,52,13,12,43,50,4,11,26,58,49,10,17,18,53,54,44,5,80,14,15,30,62]
    match_str = [str(lbl) for lbl in BRAINSTEM_LABELS]

    run_cmd([
        "mri_binarize",
        "--i", tmp_nii,
        "--match"
    ] + match_str + [
        "--inv",
        "--o", os.path.join(tmp_dir, "bin_tmp_mni.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", tmp_nii, "-mul",
        os.path.join(tmp_dir, "bin_tmp_mni.nii.gz"),
        os.path.join(tmp_dir, "brainstem_mask_mni.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", os.path.join(tmp_dir, "brainstem_mask_mni.nii.gz"),
        "-bin", final_mask_nii
    ], verbose=verbose)

    out_gii = os.path.join(tmp_dir, "brainstem_in_mni.surf.gii")
    volume_to_gifti(final_mask_nii, out_gii, level=0.5)
    return out_gii


# ===========================================================================
# 7) PROJECT PARAM MAP AND COLOR SURFACES
# ===========================================================================
def project_param_to_surface(mesh, param_nifti_path, num_colors=6, order=0):
    """
    Sample the param volume at each vertex of 'mesh'. Discretize the values 
    into 'num_colors' steps, and assign those colors to the mesh vertices.

    'order=0' => nearest neighbor interpolation (useful if your param map 
    is a labeled volume).
    'order=1' => trilinear interpolation (for continuous data).
    """
    print(f"[INFO] Projecting param map '{os.path.basename(param_nifti_path)}' -> mesh vertices")
    param_img   = nib.load(param_nifti_path)
    param_data  = param_img.get_fdata()
    param_aff   = param_img.affine

    # Transform each vertex into voxel coordinates
    inv_aff = np.linalg.inv(param_aff)
    verts = mesh.vertices
    ones  = np.ones((verts.shape[0], 1))
    vert_hom = np.hstack([verts, ones])
    vox_hom  = (inv_aff @ vert_hom.T).T
    vox_xyz  = vox_hom[:, :3]

    # Sample the volume at those voxel coords
    param_vals = map_coordinates(
        param_data,
        [vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]],
        order=order,
        mode='constant',
        cval=0.0
    )

    # Discretize the param values => color indices
    pmin, pmax = np.min(param_vals), np.max(param_vals)
    if pmin == pmax:
        color_indices = np.zeros_like(param_vals, dtype=int)
    else:
        bins = np.linspace(pmin, pmax, num_colors + 1)
        color_indices = np.digitize(param_vals, bins) - 1
        color_indices[color_indices < 0] = 0
        color_indices[color_indices >= num_colors] = num_colors - 1

    # Use the newer matplotlib colormaps API
    cmap = matplotlib.colormaps.get_cmap('viridis').resampled(num_colors)
    color_table = (cmap(range(num_colors))[:, :4] * 255).astype(np.uint8)
    vertex_colors = color_table[color_indices]
    mesh.visual.vertex_colors = vertex_colors

    return mesh


def color_pial_from_midthickness(pial_mesh_file, mid_mesh_file, param_nifti_path, num_colors=6, order=0):
    """
    1) Load midthickness surface (same # of vertices as pial).
    2) Project param map onto mid surface.
    3) Load pial surface, copy color from mid => pial (1:1).
    4) Return the colored pial mesh.
    """
    print(f"[INFO] color_pial_from_midthickness => {os.path.basename(pial_mesh_file)}")
    mid_mesh = gifti_to_trimesh(mid_mesh_file)
    mid_mesh_colored = project_param_to_surface(mid_mesh, param_nifti_path, num_colors=num_colors, order=order)

    pial_mesh = gifti_to_trimesh(pial_mesh_file)
    if len(mid_mesh_colored.vertices) != len(pial_mesh.vertices):
        raise ValueError("Midthickness & pial do NOT have matching # of vertices. Cannot copy color 1:1!")

    pial_mesh.visual.vertex_colors = mid_mesh_colored.visual.vertex_colors.copy()
    return pial_mesh


def copy_vertex_colors(mesh_source, mesh_target):
    """
    Copy per-vertex color from 'mesh_source' to 'mesh_target', 
    assuming both have the same vertex count and ordering.
    """
    if len(mesh_source.vertices) != len(mesh_target.vertices):
        raise ValueError("Source & target do NOT have matching # of vertices. Cannot copy color.")
    mesh_target.visual.vertex_colors = mesh_source.visual.vertex_colors.copy()


# ===========================================================================
# 8) MAIN
# ===========================================================================
def main():
    """
    Main entry point. Parses arguments, generates T1 or MNI surfaces, 
    optionally colors them, and exports as STL/OBJ. 
    """
    parser = argparse.ArgumentParser(
        description="Generate T1 and/or MNI surfaces (cortex+brainstem), "
                    "with optional hole-filling/smoothing. If a param_map "
                    "is given, you can color the surfaces. If --use_midthickness "
                    "is set, the script will sample the param map on the mid "
                    "surface and copy that color to the pial."
    )
    parser.add_argument("--subjects_dir", required=True,
        help="Path to the 'derivatives' folder containing each subject's anat directory.")
    parser.add_argument("--subject_id", required=True,
        help="Subject ID that also matches the naming convention in your derivatives.")
    parser.add_argument("--surfaces", choices=["both","mni","t1"], default="both",
        help="Which surfaces to generate: T1 only, MNI only, or both (default).")
    parser.add_argument("--out_warp", default="warp.nii",
        help="Name of the 4D warp field to be created (if MNI surfaces).")
    parser.add_argument("--output_dir", default=".",
        help="Where to store the output files (STL, OBJ, etc.).")
    parser.add_argument("--no_fill", action="store_true",
        help="Skip hole-filling in the extracted brainstem meshes.")
    parser.add_argument("--no_smooth", action="store_true",
        help="Skip Taubin smoothing for the extracted brainstem meshes.")
    parser.add_argument("--no_clean", action="store_true",
        help="If set, do NOT remove the temporary work folder at the end.")
    parser.add_argument("--param_map", default=None,
        help="Path to a param volume (usually in T1 space). If only MNI surfaces "
             "are requested, it will be warped to MNI. Otherwise, T1 surfaces are "
             "colored directly, and if MNI surfaces are also generated, the T1's "
             "vertex colors are copied to MNI.")
    parser.add_argument("--num_colors", type=int, default=6,
        help="Number of discrete color steps for the param map.")
    parser.add_argument("--export_obj", action="store_true",
        help="If set, produce a colored OBJ (with vertex colors) when param_map is provided.")
    parser.add_argument("--use_midthickness", action="store_true",
        help="If set, sample the param map on the mid surface (1:1 vertex match with pial).")
    parser.add_argument("--verbose", action="store_true",
        help="If set, print external commands and their outputs (default=quiet).")

    args = parser.parse_args()

    subjects_dir = args.subjects_dir
    subject_id   = args.subject_id
    surfaces     = args.surfaces
    out_warp     = args.out_warp
    output_dir   = args.output_dir
    do_fill      = not args.no_fill
    do_smooth    = not args.no_smooth
    do_clean     = not args.no_clean

    param_map    = args.param_map
    num_colors   = args.num_colors
    do_obj       = args.export_obj
    use_mid      = args.use_midthickness
    verbose      = args.verbose

    tmp_dir_name = f"_tmp_processing_{uuid.uuid4().hex[:6]}"
    tmp_dir      = os.path.join(output_dir, tmp_dir_name)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[INFO] Temporary folder created: {tmp_dir}")

    anat_dir = os.path.join(subjects_dir, subject_id, "anat")

    # Identify pial surfaces
    lh_pial_pattern = f"{anat_dir}/*_run-01_hemi-L_pial.surf.gii"
    rh_pial_pattern = f"{anat_dir}/*_run-01_hemi-R_pial.surf.gii"
    lh_pial_file    = first_match(lh_pial_pattern)
    rh_pial_file    = first_match(rh_pial_pattern)

    # If mid surfaces are needed for param sampling
    lh_mid_file = None
    rh_mid_file = None
    if use_mid:
        lh_mid_pattern = f"{anat_dir}/*_run-01_hemi-L_midthickness.surf.gii"
        rh_mid_pattern = f"{anat_dir}/*_run-01_hemi-R_midthickness.surf.gii"
        lh_mid_file    = first_match(lh_mid_pattern)
        rh_mid_file    = first_match(rh_mid_pattern)

    do_mni = (surfaces in ["both","mni"])
    do_t1  = (surfaces in ["both","t1"])

    # -----------------------------------------------------------------------
    # A) BUILD T1 SURFACES (LH + RH + brainstem)
    # -----------------------------------------------------------------------
    t1_mesh_final = None
    lh_mesh_t1    = None
    rh_mesh_t1    = None
    st_mesh_t1    = None

    if do_t1:
        print("\n[STEP] Creating T1-space surfaces (LH + RH + Brainstem)")
        brainstem_t1_gii = extract_brainstem_in_t1(subjects_dir, subject_id, tmp_dir=tmp_dir, verbose=verbose)

        # Load the GIFTIs as trimesh
        lh_mesh_t1 = gifti_to_trimesh(lh_pial_file)
        rh_mesh_t1 = gifti_to_trimesh(rh_pial_file)
        st_mesh_t1 = gifti_to_trimesh(brainstem_t1_gii)

        # Repair the brainstem mesh if requested
        if do_fill:
            print("[INFO] Filling holes in T1 brainstem mesh ...")
            trimesh.repair.fill_holes(st_mesh_t1)
        if do_smooth:
            print("[INFO] Smoothing T1 brainstem mesh (Taubin) ...")
            trimesh.smoothing.filter_taubin(st_mesh_t1, lamb=0.5, nu=-0.53, iterations=10)

        st_mesh_t1.invert()  # ensure outward normals

        # Combine LH, RH, Brainstem
        t1_mesh_final = lh_mesh_t1 + rh_mesh_t1 + st_mesh_t1
        out_stl_t1 = os.path.join(output_dir, f"{subject_id}_combined_brain_LH_RH_stem_T1.stl")
        t1_mesh_final.export(out_stl_t1, file_type='stl')
        print(f"[INFO] T1 combined STL => {os.path.basename(out_stl_t1)}")

        # If we have a param_map, color the T1 surfaces
        if param_map is not None and os.path.exists(param_map):
            print("\n[STEP] Coloring T1 surfaces ...")
            if use_mid:
                print("[INFO] Sampling param on midthickness, then copying to pial")
                # LH
                lh_colored = color_pial_from_midthickness(lh_pial_file, lh_mid_file, param_map, num_colors=num_colors)
                # RH
                rh_colored = color_pial_from_midthickness(rh_pial_file, rh_mid_file, param_map, num_colors=num_colors)
            else:
                print("[INFO] Sampling param directly on pial")
                lh_colored = gifti_to_trimesh(lh_pial_file)
                lh_colored = project_param_to_surface(lh_colored, param_map, num_colors=num_colors)

                rh_colored = gifti_to_trimesh(rh_pial_file)
                rh_colored = project_param_to_surface(rh_colored, param_map, num_colors=num_colors)

            # Re-assemble T1 with color
            colored_t1 = lh_colored + rh_colored + st_mesh_t1
            if do_obj:
                out_obj_t1 = os.path.join(output_dir, f"{subject_id}_combined_brain_LH_RH_stem_T1_colored.obj")
                colored_t1.export(out_obj_t1, file_type="obj")
                print(f"[INFO] Exported T1 colored OBJ => {os.path.basename(out_obj_t1)}")

            # Keep the colored version for later use
            t1_mesh_final = colored_t1

    # -----------------------------------------------------------------------
    # B) BUILD MNI SURFACES (LH + RH + brainstem)
    # -----------------------------------------------------------------------
    mni_mesh_final = None
    lh_mni_mesh    = None
    rh_mni_mesh    = None
    st_mni_mesh    = None

    if do_mni:
        print("\n[STEP] Creating MNI-space surfaces (via warp)")
        # Generate the warp field
        mni_file_pattern = f"{anat_dir}/*_run-01_space-MNI152NLin2009cAsym_*_T1w.nii.gz"
        t1_file_pattern  = f"{anat_dir}/*_run-01_desc-preproc_T1w.nii.gz"
        xfm_pattern      = f"{anat_dir}/*_run-01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"

        mni_file = first_match(mni_file_pattern)
        t1_file  = first_match(t1_file_pattern)
        xfm_file = first_match(xfm_pattern)

        generate_mrtrix_style_warp(
            mni_file=mni_file,
            t1_file=t1_file,
            xfm_file=xfm_file,
            out_warp=out_warp,
            tmp_dir=tmp_dir,
            verbose=verbose
        )

        # Warp the pial surfaces
        lh_mni_out = os.path.join(tmp_dir, f"{subject_id}_run-01_hemi-L_pial_MNI.surf.gii")
        rh_mni_out = os.path.join(tmp_dir, f"{subject_id}_run-01_hemi-R_pial_MNI.surf.gii")
        warp_gifti_vertices(lh_pial_file, os.path.join(tmp_dir, out_warp), lh_mni_out, verbose=verbose)
        warp_gifti_vertices(rh_pial_file, os.path.join(tmp_dir, out_warp), rh_mni_out, verbose=verbose)

        # If using mid, also warp the mid thickness surfaces
        if use_mid:
            lh_mni_mid_out = os.path.join(tmp_dir, f"{subject_id}_run-01_hemi-L_midthickness_MNI.surf.gii")
            rh_mni_mid_out = os.path.join(tmp_dir, f"{subject_id}_run-01_hemi-R_midthickness_MNI.surf.gii")
            warp_gifti_vertices(lh_mid_file, os.path.join(tmp_dir, out_warp), lh_mni_mid_out, verbose=verbose)
            warp_gifti_vertices(rh_mid_file, os.path.join(tmp_dir, out_warp), rh_mni_mid_out, verbose=verbose)

        # Brainstem in MNI
        out_aseg_in_mni = os.path.join(tmp_dir, "aseg_in_mni.nii.gz")
        brainstem_mni   = extract_brainstem_in_mni(subjects_dir, subject_id, out_aseg_in_mni, tmp_dir=tmp_dir, verbose=verbose)

        lh_mni_mesh = gifti_to_trimesh(lh_mni_out)
        rh_mni_mesh = gifti_to_trimesh(rh_mni_out)
        st_mni_mesh = gifti_to_trimesh(brainstem_mni)

        if do_fill:
            print("[INFO] Filling holes in MNI brainstem mesh ...")
            trimesh.repair.fill_holes(st_mni_mesh)
        if do_smooth:
            print("[INFO] Smoothing MNI brainstem mesh (Taubin) ...")
            trimesh.smoothing.filter_taubin(st_mni_mesh, lamb=0.5, nu=-0.53, iterations=10)

        st_mni_mesh.invert()

        combined_mni = lh_mni_mesh + rh_mni_mesh + st_mni_mesh
        out_stl_mni  = os.path.join(output_dir, f"{subject_id}_combined_brain_LH_RH_stem_MNI.stl")
        combined_mni.export(out_stl_mni, file_type='stl')
        print(f"[INFO] MNI combined STL => {os.path.basename(out_stl_mni)}")

        mni_mesh_final = combined_mni

        # -------------------------------------------------------------------
        # Coloring the MNI surfaces:
        # If T1 was created and param_map was given, we can just copy T1's color.
        # Otherwise, warp the param_map to MNI and project it directly.
        # -------------------------------------------------------------------
        if param_map is not None and os.path.exists(param_map):
            if do_t1 and t1_mesh_final is not None:
                print("\n[STEP] Copying T1 vertex colors to MNI surfaces")

                # We re-color the T1 pial surfaces for LH & RH individually:
                print("[INFO] Re-coloring T1 pial surfaces so we can copy them to MNI")
                if use_mid:
                    lh_colored = color_pial_from_midthickness(lh_pial_file, lh_mid_file, param_map, num_colors=num_colors)
                    rh_colored = color_pial_from_midthickness(rh_pial_file, rh_mid_file, param_map, num_colors=num_colors)
                else:
                    lh_colored = gifti_to_trimesh(lh_pial_file)
                    lh_colored = project_param_to_surface(lh_colored, param_map, num_colors=num_colors)
                    rh_colored = gifti_to_trimesh(rh_pial_file)
                    rh_colored = project_param_to_surface(rh_colored, param_map, num_colors=num_colors)

                # Copy color from T1 -> MNI 
                copy_vertex_colors(lh_colored, lh_mni_mesh)
                copy_vertex_colors(rh_colored, rh_mni_mesh)

                # Combine MNI again
                colored_mni = lh_mni_mesh + rh_mni_mesh + st_mni_mesh
                if do_obj:
                    out_obj_mni = os.path.join(output_dir, f"{subject_id}_combined_brain_LH_RH_stem_MNI_colored.obj")
                    colored_mni.export(out_obj_mni, file_type="obj")
                    print(f"[INFO] Exported MNI colored OBJ => {os.path.basename(out_obj_mni)}")

                mni_mesh_final = colored_mni

            else:
                print("\n[STEP] Warping param_map to MNI and projecting it directly")
                
                colored_mni = project_param_to_surface(mni_mesh_final, param_map, num_colors=num_colors)
                if do_obj:
                    out_obj_mni = os.path.join(output_dir, f"{subject_id}_combined_brain_LH_RH_stem_MNI_colored.obj")
                    colored_mni.export(out_obj_mni, file_type="obj")
                    print(f"[INFO] Exported MNI colored OBJ => {os.path.basename(out_obj_mni)}")

                mni_mesh_final = colored_mni

    # -----------------------------------------------------------------------
    # C) CLEANUP
    # -----------------------------------------------------------------------
    if do_clean:
        print(f"\n[INFO] Cleaning up: removing temporary folder => {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")


if __name__ == "__main__":
    main()

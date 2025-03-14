# brain_for_printing/cli_mni.py

import os
import argparse
import uuid
import shutil
import trimesh

from .io_utils import first_match
from .mesh_utils import gifti_to_trimesh
from .warp_utils import generate_mrtrix_style_warp, warp_gifti_vertices
from .surfaces import extract_brainstem_in_mni
from .color_utils import (
    project_param_to_surface,
    color_pial_from_midthickness,
    copy_vertex_colors
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate MNI-space surfaces (LH pial, RH pial, optional brainstem). "
                    "Optionally color the surfaces."
    )
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--out_warp", default="warp.nii",
        help="Name of the 4D warp field to create.")
    parser.add_argument("--no_brainstem", action="store_true")
    parser.add_argument("--no_fill", action="store_true")
    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--param_map", default=None)
    parser.add_argument("--use_midthickness", action="store_true")
    parser.add_argument("--num_colors", type=int, default=6)
    parser.add_argument("--export_obj", action="store_true")
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    do_fill = not args.no_fill
    do_smooth = not args.no_smooth
    do_clean = not args.no_clean

    # Temporary folder
    tmp_dir = os.path.join(args.output_dir, f"_tmp_mni_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[INFO] Temporary folder => {tmp_dir}")

    anat_dir = os.path.join(args.subjects_dir, args.subject_id, "anat")

    # 1) Identify required files
    lh_pial_pattern = f"{anat_dir}/*_run-01_hemi-L_pial.surf.gii"
    rh_pial_pattern = f"{anat_dir}/*_run-01_hemi-R_pial.surf.gii"
    lh_pial_file = first_match(lh_pial_pattern)
    rh_pial_file = first_match(rh_pial_pattern)

    # (Optional) Mid surfaces
    lh_mid_file = rh_mid_file = None
    if args.use_midthickness:
        lh_mid_pattern = f"{anat_dir}/*_run-01_hemi-L_midthickness.surf.gii"
        rh_mid_pattern = f"{anat_dir}/*_run-01_hemi-R_midthickness.surf.gii"
        lh_mid_file = first_match(lh_mid_pattern)
        rh_mid_file = first_match(rh_mid_pattern)

    # 2) Create warp (MNI -> T1)
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
        out_warp=args.out_warp,
        tmp_dir=tmp_dir,
        verbose=args.verbose
    )

    # 3) Warp LH & RH pial
    lh_mni_out = os.path.join(tmp_dir, f"{args.subject_id}_L_pial_MNI.surf.gii")
    rh_mni_out = os.path.join(tmp_dir, f"{args.subject_id}_R_pial_MNI.surf.gii")
    warp_gifti_vertices(lh_pial_file, os.path.join(tmp_dir, args.out_warp),
                        lh_mni_out, verbose=args.verbose)
    warp_gifti_vertices(rh_pial_file, os.path.join(tmp_dir, args.out_warp),
                        rh_mni_out, verbose=args.verbose)

    # (Optional) Warp mid
    lh_mni_mid_out = rh_mni_mid_out = None
    if args.use_midthickness:
        lh_mni_mid_out = os.path.join(tmp_dir, f"{args.subject_id}_L_mid_MNI.surf.gii")
        rh_mni_mid_out = os.path.join(tmp_dir, f"{args.subject_id}_R_mid_MNI.surf.gii")
        warp_gifti_vertices(lh_mid_file, os.path.join(tmp_dir, args.out_warp),
                            lh_mni_mid_out, verbose=args.verbose)
        warp_gifti_vertices(rh_mid_file, os.path.join(tmp_dir, args.out_warp),
                            rh_mni_mid_out, verbose=args.verbose)

    # 4) Extract brainstem in MNI if requested
    st_mni_mesh = None
    if not args.no_brainstem:
        out_aseg_in_mni = os.path.join(tmp_dir, "aseg_in_mni.nii.gz")
        bs_mni_gii = extract_brainstem_in_mni(
            subjects_dir=args.subjects_dir,
            subject_id=args.subject_id,
            out_aseg_in_mni=out_aseg_in_mni,
            tmp_dir=tmp_dir,
            verbose=args.verbose
        )
        st_mni_mesh = gifti_to_trimesh(bs_mni_gii)
        if do_fill:
            trimesh.repair.fill_holes(st_mni_mesh)
        if do_smooth:
            trimesh.smoothing.filter_taubin(st_mni_mesh, lamb=0.5, nu=-0.53, iterations=10)
        st_mni_mesh.invert()

    # 5) Combine LH/RH (+ brainstem)
    lh_mni_mesh = gifti_to_trimesh(lh_mni_out)
    rh_mni_mesh = gifti_to_trimesh(rh_mni_out)
    if st_mni_mesh:
        mni_mesh_final = lh_mni_mesh + rh_mni_mesh + st_mni_mesh
    else:
        mni_mesh_final = lh_mni_mesh + rh_mni_mesh

    out_stl_mni = os.path.join(args.output_dir,
                               f"{args.subject_id}_MNI_brain.stl")
    mni_mesh_final.export(out_stl_mni, file_type="stl")
    print(f"[INFO] Exported => {out_stl_mni}")

    # 6) (Optional) color
    if args.param_map:
        # if we want to sample param_map from mid or from T1
        if args.use_midthickness and lh_mni_mid_out and rh_mni_mid_out:
            # Sample param_map on warped MNI mid surfaces, then copy to pial
            lh_mni_mid_mesh = gifti_to_trimesh(lh_mni_mid_out)
            rh_mni_mid_mesh = gifti_to_trimesh(rh_mni_mid_out)

            lh_mni_mid_colored = project_param_to_surface(lh_mni_mid_mesh,
                                                          args.param_map,
                                                          num_colors=args.num_colors)
            rh_mni_mid_colored = project_param_to_surface(rh_mni_mid_mesh,
                                                          args.param_map,
                                                          num_colors=args.num_colors)

            copy_vertex_colors(lh_mni_mid_colored, lh_mni_mesh)
            copy_vertex_colors(rh_mni_mid_colored, rh_mni_mesh)

            if st_mni_mesh:
                colored_mni = lh_mni_mesh + rh_mni_mesh + st_mni_mesh
            else:
                colored_mni = lh_mni_mesh + rh_mni_mesh

        else:
            # Direct param map projection on the combined MNI mesh
            colored_mni = project_param_to_surface(mni_mesh_final,
                                                   args.param_map,
                                                   num_colors=args.num_colors)

        if args.export_obj:
            out_obj_mni = os.path.join(args.output_dir,
                                       f"{args.subject_id}_MNI_colored.obj")
            colored_mni.export(out_obj_mni, file_type="obj")
            print(f"[INFO] Exported => {out_obj_mni}")

    # 7) Cleanup
    if do_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")


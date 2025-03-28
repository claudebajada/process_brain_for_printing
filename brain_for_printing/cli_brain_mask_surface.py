# brain_for_printing/cli_brain_mask_surface.py

import os
import argparse
import uuid
import shutil
import trimesh
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure

from .io_utils import first_match, run_cmd
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .log_utils import write_log

def main():
    parser = argparse.ArgumentParser(
        description="Generate a smooth (optionally inflated) surface from a T1w brain mask, optionally warp to MNI."
    )
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--space", choices=["T1", "MNI"], default="T1")
    parser.add_argument("--inflate_mm", type=float, default=0.0,
        help="Amount to inflate the brain mask in mm before creating the surface.")
    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--no_clean", action="store_true")
    args = parser.parse_args()

    do_smooth = not args.no_smooth
    do_clean = not args.no_clean

    tmp_dir = os.path.join(args.output_dir, f"_tmp_masksurf_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)

    log = {
        "tool": "brain_for_printing_brain_mask_surface",
        "subject_id": args.subject_id,
        "output_dir": args.output_dir,
        "space": args.space,
        "inflate_mm": args.inflate_mm,
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    anat_dir = os.path.join(args.subjects_dir, args.subject_id, "anat")
    mask_pattern = f"{anat_dir}/*_run-01_desc-brain_mask.nii.gz"
    mask_file = first_match(mask_pattern)
    log["mask_file"] = mask_file
    log["steps"].append("Located T1w brain mask")

    # Warp to MNI if needed
    final_mask = mask_file
    if args.space == "MNI":
        xfm_pattern = f"{anat_dir}/*_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
        mni_template_pattern = f"{anat_dir}/*_run-01_space-MNI152NLin2009cAsym_*_T1w.nii.gz"
        xfm_file = first_match(xfm_pattern)
        mni_template = first_match(mni_template_pattern)

        final_mask = os.path.join(tmp_dir, "brain_mask_mni.nii.gz")
        run_cmd([
            "antsApplyTransforms", "-d", "3",
            "-i", mask_file,
            "-o", final_mask,
            "-r", mni_template,
            "-t", xfm_file,
            "-n", "NearestNeighbor"
        ])
        log["steps"].append("Warped brain mask to MNI space")

    # Inflate the binary mask if requested
    inflated_mask_nii = os.path.join(tmp_dir, "inflated_mask.nii.gz")
    nii = nib.load(final_mask)
    mask_data = nii.get_fdata() > 0
    affine = nii.affine

    if args.inflate_mm > 0.0:
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        avg_voxel_mm = np.mean(voxel_sizes)
        n_voxels = int(np.ceil(args.inflate_mm / avg_voxel_mm))

        struct = generate_binary_structure(3, 1)
        dilated = mask_data.copy()
        for _ in range(n_voxels):
            dilated = binary_dilation(dilated, structure=struct)

        inflated_img = nib.Nifti1Image(dilated.astype(np.uint8), affine)
        nib.save(inflated_img, inflated_mask_nii)
        log["steps"].append(f"Inflated binary mask by {args.inflate_mm} mm (~{n_voxels} voxels)")
        final_mask = inflated_mask_nii

    # Convert to GIFTI
    out_gii = os.path.join(tmp_dir, "brain_mask_surface.gii")
    volume_to_gifti(final_mask, out_gii, level=0.5)
    log["steps"].append("Converted brain mask to GIFTI")

    # Load and smooth + invert
    mesh = gifti_to_trimesh(out_gii)
    if do_smooth:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
        log["steps"].append("Applied Taubin smoothing")

    mesh.invert()
    log["steps"].append("Inverted mesh normals")

    # Export STL
    out_stl = os.path.join(args.output_dir, f"{args.subject_id}_{args.space}_mask_surface.stl")
    mesh.export(out_stl, file_type="stl")
    log["output_files"].append(out_stl)
    log["steps"].append("Exported STL mesh")
    log["vertices"] = len(mesh.vertices)

    # Write log
    write_log(log, args.output_dir, base_name="mask_surface_log")

    if not do_clean:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()


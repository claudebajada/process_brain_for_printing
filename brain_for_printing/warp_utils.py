# brain_for_printing/warp_utils.py

import os
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path
from scipy.ndimage import map_coordinates
import logging # Use logging
import sys # Import sys for potential exits
import uuid # Keep Added Import

from .io_utils import run_cmd # Assuming run_cmd handles errors properly

L = logging.getLogger(__name__)

def create_mrtrix_warp(
    mni_template_file: str, # Input MNI template (defines the grid for warpinit)
    t1_native_file: str,    # Input T1w file (defines target geometry for antsApplyTransforms -r)
    h5_transform_mni_to_t1: str, # ANTs H5 transform mapping points FROM MNI TO T1
    output_warp_file: str,  # Output: 4D warp field (MNI grid -> T1 coords)
    tmp_dir: str = ".",
    verbose: bool = False
):
    """
    Generate a 4D (X, Y, Z, 3) warp field in MRtrix format that stores,
    for each voxel in the MNI grid, the corresponding coordinate in T1 space.
    This follows the logic often needed for MRtrix coordinate transformations.

    Args:
        mni_template_file: Path to NIfTI MNI template (defines warp field grid).
        t1_native_file: Path to NIfTI T1w image (defines target geometry for ANTs).
        h5_transform_mni_to_t1: Path to ANTs H5 transform file (MNI -> T1).
        output_warp_file: Path to save the output 4D warp NIfTI file.
        tmp_dir: Temporary directory for intermediate files.
        verbose: Enable verbose command output.
    """
    L.info(f"Generating MRtrix-style MNI->T1 coordinate warp field.")
    L.info(f"  Grid Reference (warpinit): {Path(mni_template_file).name}")
    L.info(f"  Target Geometry (ANTs -r): {Path(t1_native_file).name}")
    L.info(f"  Transform (ANTs -t): {Path(h5_transform_mni_to_t1).name}")
    L.info(f"  Output Warp Field: {output_warp_file}")

    # Input file checks
    mni_path = Path(mni_template_file)
    t1_path = Path(t1_native_file)
    h5_path = Path(h5_transform_mni_to_t1)
    if not mni_path.exists(): L.error(f"MNI template file not found: {mni_path}"); raise FileNotFoundError(f"MNI template file not found: {mni_path}")
    if not t1_path.exists(): L.error(f"T1 native file not found: {t1_path}"); raise FileNotFoundError(f"T1 native file not found: {t1_path}")
    if not h5_path.exists(): L.error(f"H5 transform (MNI->T1) not found: {h5_path}"); raise FileNotFoundError(f"H5 transform (MNI->T1) not found: {h5_path}")

    tmp_dir_path = Path(tmp_dir)
    unique_id = uuid.uuid4().hex[:6]
    base_name = f"{Path(output_warp_file).stem.replace('.nii','').replace('.gz','')}_{unique_id}"
    
    # 1) Initialize empty warp images using MNI template
    no_warp_template = tmp_dir_path / f"{base_name}_no-warp-[].nii"
    cmd_init = ["warpinit", str(mni_path), str(no_warp_template)]
    L.debug(f"Running warpinit: {' '.join(cmd_init)}")
    try:
        run_cmd(cmd_init, verbose=verbose)
    except Exception as e_init:
        L.error(f"warpinit command failed: {e_init}")
        raise

    # 2) Apply transforms for each dimension
    warped_grids = []
    try:
        for i in range(3):
            no_warp_i = tmp_dir_path / f"{base_name}_no-warp-{i}.nii"
            warp_i = tmp_dir_path / f"{base_name}_warp_{i}.nii"
            cmd_apply = [
                "antsApplyTransforms", "-d", "3",
                "-i", str(no_warp_i),
                "-r", str(t1_path),
                "-o", str(warp_i),
                "-t", str(h5_path),
                "-n", "Linear"
            ]
            L.debug(f"Running antsApplyTransforms (dim {i}): {' '.join(cmd_apply)}")
            run_cmd(cmd_apply, verbose=verbose)
            if not warp_i.exists():
                L.error(f"antsApplyTransforms failed: Output warped grid not found: {warp_i}")
                raise RuntimeError(f"antsApplyTransforms failed for dimension {i}")
            warped_grids.append(str(warp_i))
        L.info("antsApplyTransforms completed successfully (MNI grid -> T1 coords).")
    except Exception as e_apply:
        L.error(f"antsApplyTransforms failed: {e_apply}")
        # Clean up intermediate files
        for p in tmp_dir_path.glob(f"{base_name}_no-warp-*.nii"): p.unlink(missing_ok=True)
        for p in tmp_dir_path.glob(f"{base_name}_warp_*.nii"): p.unlink(missing_ok=True)
        raise

    # 3) Concatenate the 3 warped coordinate images into a single 4D warp field
    try:
        cmd_cat = ["mrcat", *warped_grids, str(output_warp_file), "-axis", "3"]
        L.debug(f"Running mrcat: {' '.join(cmd_cat)}")
        run_cmd(cmd_cat, verbose=verbose)
        L.info("mrcat completed successfully.")
    except Exception as e_cat:
        L.error(f"mrcat failed: {e_cat}")
        # Clean up intermediate files
        for p in tmp_dir_path.glob(f"{base_name}_no-warp-*.nii"): p.unlink(missing_ok=True)
        for p in tmp_dir_path.glob(f"{base_name}_warp_*.nii"): p.unlink(missing_ok=True)
        raise

    # Check final output
    output_path = Path(output_warp_file)
    if not output_path.exists() or output_path.stat().st_size == 0:
        L.error(f"mrcat check failed: Final warp field empty or not created: {output_path}")
        # Clean up intermediate files
        for p in tmp_dir_path.glob(f"{base_name}_no-warp-*.nii"): p.unlink(missing_ok=True)
        for p in tmp_dir_path.glob(f"{base_name}_warp_*.nii"): p.unlink(missing_ok=True)
        raise RuntimeError(f"mrcat failed to create the final warp field: {output_path}")

    L.info(f"Successfully created MRtrix-style MNI->T1 warp => {output_warp_file}")

    # Clean up intermediate files
    L.debug("Cleaning up intermediate warp files...")
    for p in tmp_dir_path.glob(f"{base_name}_no-warp-*.nii"): p.unlink(missing_ok=True)
    for p in tmp_dir_path.glob(f"{base_name}_warp_*.nii"): p.unlink(missing_ok=True)


# --- generate_mrtrix_style_warp (Legacy wrapper) ---
# This now correctly calls the revised create_mrtrix_warp
def generate_mrtrix_style_warp(mni_file, t1_file, xfm_mni_to_t1_file, output_warp_file="warp.nii", tmp_dir=".", verbose=False):
    L.warning("Using legacy function generate_mrtrix_style_warp. Use create_mrtrix_warp instead.")
    create_mrtrix_warp(
        mni_template_file=mni_file,
        t1_native_file=t1_file,
        h5_transform_mni_to_t1=xfm_mni_to_t1_file,
        output_warp_file=output_warp_file,
        tmp_dir=tmp_dir,
        verbose=verbose
    )


# --- warp_gifti_vertices ---
# WARNING: This function expects a warp field that maps coordinates FROM
# the GIFTI's current space TO the target space. The warp field generated
# by the modified create_mrtrix_warp above maps MNI -> T1. Applying it
# directly to T1 vertices using this function will NOT result in MNI coordinates.
def warp_gifti_vertices(gifti_file, warp_field_file, output_gifti_file, verbose=False):
    L.info(f"Warping GIFTI: {Path(gifti_file).name} using {Path(warp_field_file).name}")
    try:
        gifti_img = nib.load(gifti_file); verts_d = gifti_img.darrays[0]; faces_d = gifti_img.darrays[1]; verts = verts_d.data.astype(np.float64)
        warp_img = nib.load(warp_field_file); warp_data = warp_img.get_fdata(dtype=np.float64); inv_affine = np.linalg.inv(warp_img.affine)
        ones = np.ones((verts.shape[0], 1), dtype=verts.dtype); vert_hom = np.hstack([verts, ones])
        # Convert vertices to voxel coordinates of the warp field grid
        vox_coords_warp = (inv_affine @ vert_hom.T)[:3]
        warped_coords = np.zeros_like(verts, dtype=np.float64)
        # Sample the warp field at these coordinates
        # This assumes warp_data[..., dim] contains the target coordinate for that dimension
        for dim in range(3): warped_coords[:, dim] = map_coordinates(warp_data[..., dim], vox_coords_warp, order=1, mode='nearest')
        # Create new GIFTI
        warped_verts_da = nib.gifti.GiftiDataArray( data=warped_coords.astype(np.float32), intent=verts_d.intent, datatype=verts_d.datatype, meta=verts_d.meta )
        faces_da_copy = nib.gifti.GiftiDataArray( data=faces_d.data, intent=faces_d.intent, datatype=faces_d.datatype, meta=faces_d.meta )
        out_gii = nib.gifti.GiftiImage(darrays=[warped_verts_da, faces_da_copy], meta=gifti_img.meta); nib.save(out_gii, output_gifti_file)
        L.info(f"Saved warped GIFTI => {Path(output_gifti_file).name}")
    except FileNotFoundError as e: L.error(f"GIFTI warping FileNotFoundError: {e}"); raise
    except Exception as e: L.error(f"GIFTI warping error for {Path(gifti_file).name}: {e}", exc_info=verbose); raise



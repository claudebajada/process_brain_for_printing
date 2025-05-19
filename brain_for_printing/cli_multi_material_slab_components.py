#!/usr/bin/env python
# brain_for_printing/cli_multi_material_slab_components.py

"""
Generates distinct, non-overlapping multi-material brain slab components
suitable for 3D printing.Workflow:
1. AC-PC align T1w.
2. Generate parent surfaces (BrainMask, Pial complex, White complex, SGM/Ventricles) in T1w space.
3. Voxelize parent T1w surfaces onto a high-resolution AC-PC grid (transforming them to AC-PC space first).
4. Crop these full AC-PC binarized masks.
5. Extract volumetric slabs from the cropped AC-PC masks.
6. Perform volumetric math on these AC-PC slabs to define material NIfTIs.
7. Resample these AC-PC material NIfTIs back to native T1w space.
8. Mesh these native T1w material NIfTIs to create initial native T1w space STLs.
9. Optionally, flatten caps:
    a. Transform native mesh to AC-PC space.
    b. Snap vertices to ideal AC-PC planes.
    c. Output either in AC-PC space (if requested) or transform back to native T1w space.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import shutil
import uuid

import trimesh
import nibabel as nib
import numpy as np
from typing import Dict, Optional, List, Any, Tuple

# --- Local Imports ---
from .io_utils import temp_dir, require_cmds, flexible_match, run_cmd
from .log_utils import get_logger, write_log
from .surfgen_utils import generate_single_brain_mask_surface, generate_brain_surfaces
from .five_tt_utils import run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes, is_vtk_available
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .config_utils import parse_preset
from .volumetric_utils import (
    regrid_to_resolution,
    mesh_to_partial_volume,
    binarize_volume_file,
    save_numpy_as_nifti,
    vol_subtract_numpy,
    vol_intersect_numpy,
    vol_union_numpy,
    load_nifti_data
)

FSLDIR_ENV = os.getenv('FSLDIR')
MNI_TEMPLATE_NAME_DEFAULT = "MNI152_T1_1mm_brain.nii.gz"
DEFAULT_MNI_TEMPLATE_PATH = Path(FSLDIR_ENV, "data", "standard", MNI_TEMPLATE_NAME_DEFAULT) if FSLDIR_ENV else None

_L_HELPERS = logging.getLogger(__name__ + ".helpers")

FSLPY_AVAILABLE = False
try:
    from fsl.data.image import Image as FSLImage
    from fsl.transform.flirt import fromFlirt as fslpy_fromFlirt
    FSLPY_AVAILABLE = True
    _L_HELPERS.info("fslpy library found and will be used for FLIRT matrix conversions.")
except ImportError:
    _L_HELPERS.warning("fslpy library not found. FLIRT matrix to world-world affine conversion will be limited. "
                     "Cap flattening relies on these for accurate transformations.")
    pass

def get_flirt_world_to_world_affine(
    fsl_flirt_mat_path: Path,
    src_image_path: Path,
    ref_image_path: Path,
    logger: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:
    local_logger = logger or _L_HELPERS
    if not fsl_flirt_mat_path.exists():
        local_logger.error(f"FLIRT matrix file not found: {fsl_flirt_mat_path}")
        return None
    if not src_image_path.exists():
        local_logger.error(f"Source image for FLIRT matrix conversion not found: {src_image_path}")
        return None
    if not ref_image_path.exists():
        local_logger.error(f"Reference image for FLIRT matrix conversion not found: {ref_image_path}")
        return None

    if not FSLPY_AVAILABLE:
        local_logger.error(
            f"fslpy is not available, which is required for accurate conversion of "
            f"FLIRT matrix {fsl_flirt_mat_path.name} to a world-to-world affine."
        )
        return None
    try:
        flirt_mat_np = np.loadtxt(str(fsl_flirt_mat_path))
        if flirt_mat_np.shape != (4, 4):
            local_logger.error(f"FLIRT matrix {fsl_flirt_mat_path.name} is not a 4x4 matrix. Shape: {flirt_mat_np.shape}")
            return None
        
        src_fslimage = FSLImage(str(src_image_path))
        ref_fslimage = FSLImage(str(ref_image_path))
        
        world_to_world_affine = fslpy_fromFlirt(
            flirt_mat_np, src_fslimage, ref_fslimage, from_='world', to='world'
        )
        local_logger.info(f"Successfully converted FLIRT matrix {fsl_flirt_mat_path.name} to world-to-world affine using fslpy.")
        return world_to_world_affine
    except Exception as e:
        local_logger.error(f"Error converting FLIRT matrix {fsl_flirt_mat_path.name} using fslpy: {e}", exc_info=True)
        return None

def get_ideal_cap_coords_world_acpc(
    slab_def_vox_start: int,
    slab_def_vox_thickness: int,
    slice_axis_idx_acpc: int,
    cropped_acpc_grid_affine: np.ndarray,
    logger: logging.Logger
) -> Optional[Tuple[float, float]]:
    try:
        min_plane_vox_coord = np.zeros(4)
        min_plane_vox_coord[slice_axis_idx_acpc] = slab_def_vox_start
        min_plane_vox_coord[3] = 1
        max_plane_vox_coord = np.zeros(4)
        max_plane_vox_coord[slice_axis_idx_acpc] = slab_def_vox_start + slab_def_vox_thickness
        max_plane_vox_coord[3] = 1
        min_plane_world_hom = cropped_acpc_grid_affine @ min_plane_vox_coord
        max_plane_world_hom = cropped_acpc_grid_affine @ max_plane_vox_coord
        ideal_coord_min_world = min_plane_world_hom[slice_axis_idx_acpc]
        ideal_coord_max_world = max_plane_world_hom[slice_axis_idx_acpc]
        if ideal_coord_min_world > ideal_coord_max_world: 
            ideal_coord_min_world, ideal_coord_max_world = ideal_coord_max_world, ideal_coord_min_world
        return float(ideal_coord_min_world), float(ideal_coord_max_world)
    except Exception as e:
        logger.error(f"Failed to calculate ideal cap coordinates: {e}", exc_info=True)
        return None

def flatten_mesh_caps_acpc(
    mesh: trimesh.Trimesh, # Expected to be in Native T1w space
    native_to_acpc_world_affine: np.ndarray,
    acpc_to_native_world_affine: np.ndarray,
    ideal_coord_min_acpc: float,
    ideal_coord_max_acpc: float,
    slice_axis_idx_acpc: int,
    tolerance_mm: float,
    logger: logging.Logger,
    output_space_is_acpc: bool = False
) -> Optional[trimesh.Trimesh]:

    if mesh.is_empty:
        logger.warning("Input mesh for flattening is empty. Skipping.")
        return mesh

    original_native_vertices = mesh.vertices.copy() # Keep a copy of original native vertices

    try:
        # Step 1: Transform input (native) mesh to ACPC space for snapping
        mesh_acpc_transformed = mesh.copy()
        logger.debug("Transforming input (native) mesh to AC-PC space for cap flattening.")
        mesh_acpc_transformed.apply_transform(native_to_acpc_world_affine)
        
        logger.debug(f"Mesh for snapping (in AC-PC space). Bounds: {mesh_acpc_transformed.bounds}")

        # Step 2: Perform snapping in ACPC space
        acpc_vertices_for_snapping = mesh_acpc_transformed.vertices.copy()
        modified_vertices_acpc = acpc_vertices_for_snapping.copy()
        target_coord_idx = slice_axis_idx_acpc

        min_cap_mask = np.abs(acpc_vertices_for_snapping[:, target_coord_idx] - ideal_coord_min_acpc) < tolerance_mm
        num_min_verts = np.sum(min_cap_mask)
        if num_min_verts > 0:
            modified_vertices_acpc[min_cap_mask, target_coord_idx] = ideal_coord_min_acpc
            logger.info(f"Snapped {num_min_verts} vertices to min AC-PC cap (IdealCoord={ideal_coord_min_acpc:.3f} on axis {target_coord_idx})")
        else:
            logger.warning(f"No vertices found within tolerance for min AC-PC cap (IdealCoord={ideal_coord_min_acpc:.3f} on axis {target_coord_idx})")

        max_cap_mask = np.abs(acpc_vertices_for_snapping[:, target_coord_idx] - ideal_coord_max_acpc) < tolerance_mm
        num_max_verts = np.sum(max_cap_mask)
        if num_max_verts > 0:
            modified_vertices_acpc[max_cap_mask, target_coord_idx] = ideal_coord_max_acpc
            logger.info(f"Snapped {num_max_verts} vertices to max AC-PC cap (IdealCoord={ideal_coord_max_acpc:.3f} on axis {target_coord_idx})")
        else:
            logger.warning(f"No vertices found within tolerance for max AC-PC cap (IdealCoord={ideal_coord_max_acpc:.3f} on axis {target_coord_idx})")

        if num_min_verts == 0 and num_max_verts == 0:
            logger.info("No vertices were snapped for either cap.")
            if output_space_is_acpc:
                logger.info("Returning original mesh transformed to ACPC space (as no snapping occurred).")
                return mesh_acpc_transformed # Already transformed to ACPC
            else:
                logger.info("Returning original native mesh (as no snapping occurred).")
                return mesh # Original native mesh

        # Create a new mesh with the snapped ACPC vertices
        flattened_mesh_acpc = trimesh.Trimesh(vertices=modified_vertices_acpc, faces=mesh.faces, process=False)

        # Step 3: Process and decide output space
        if output_space_is_acpc:
            logger.info("Processing and outputting flattened mesh in ACPC space.")
            final_processed_mesh = flattened_mesh_acpc
            final_processed_mesh.remove_degenerate_faces()
            if final_processed_mesh.is_empty:
                logger.error("Mesh (ACPC) became empty after removing degenerate faces post-flattening. Reverting to pre-snapping ACPC state.")
                return mesh_acpc_transformed.process(validate=True) if not mesh_acpc_transformed.is_empty else None

            final_processed_mesh.fix_normals(multibody=True)
            final_processed_mesh.process(validate=True)
            if final_processed_mesh.is_empty:
                logger.error("Mesh (ACPC) became empty after final processing post-flattening. Reverting to pre-snapping ACPC state.")
                return mesh_acpc_transformed.process(validate=True) if not mesh_acpc_transformed.is_empty else None
            logger.info("Successfully flattened mesh caps and processed (ACPC output).")
            return final_processed_mesh
        else: # Output in native space
            logger.info("Transforming flattened mesh back to native space and processing.")
            flattened_mesh_native = flattened_mesh_acpc.copy()
            flattened_mesh_native.apply_transform(acpc_to_native_world_affine)
            logger.debug(f"Flattened mesh transformed back to native space. Bounds: {flattened_mesh_native.bounds}")
            
            final_processed_mesh = flattened_mesh_native
            final_processed_mesh.remove_degenerate_faces()
            if final_processed_mesh.is_empty:
                 logger.error("Mesh (Native) became empty after removing degenerate faces post-flattening. Reverting to original (native).")
                 return trimesh.Trimesh(vertices=original_native_vertices, faces=mesh.faces, process=True)

            final_processed_mesh.fix_normals(multibody=True)
            final_processed_mesh.process(validate=True)
            if final_processed_mesh.is_empty:
                logger.error("Mesh (Native) became empty after final processing post-flattening. Reverting to original (native).")
                return trimesh.Trimesh(vertices=original_native_vertices, faces=mesh.faces, process=True)
            
            logger.info("Successfully flattened mesh caps and processed (Native output).")
            return final_processed_mesh

    except Exception as e:
        logger.error(f"Error during mesh cap flattening: {e}", exc_info=True)
        # Fallback: return original native mesh
        return trimesh.Trimesh(vertices=original_native_vertices, faces=mesh.faces, process=True)


def acpc_align_t1w(
    input_t1w_path: Path,
    output_acpc_t1w_path: Path,
    output_transform_mat_path: Path, # This .mat will be named based on input_t1w_path initially
    mni_template_path: Path,
    temp_dir_path: Path,
    logger: logging.Logger,
    verbose: bool = False
) -> Tuple[bool, Optional[Path]]: # Returns success and the actual path used as input to FLIRT
    logger.info(f"Starting AC-PC alignment for {input_t1w_path.name} -> {output_acpc_t1w_path.name}")
    require_cmds(["flirt", "robustfov"], logger=logger)

    if not mni_template_path.exists():
        logger.error(f"MNI template not found at {mni_template_path}.")
        return False, None

    robustfov_out_path = temp_dir_path / f"{input_t1w_path.name.replace('.nii.gz','').replace('.nii','')}_robustfov.nii.gz"
    t1w_to_flirt_actual: Path = input_t1w_path 

    try:
        logger.debug(f"Running robustfov command: {' '.join(['robustfov', '-i', str(input_t1w_path), '-r', str(robustfov_out_path)])}")
        run_cmd(["robustfov", "-i", str(input_t1w_path), "-r", str(robustfov_out_path)], verbose=verbose) 
        if robustfov_out_path.exists() and robustfov_out_path.stat().st_size > 0:
            t1w_to_flirt_actual = robustfov_out_path
            logger.info(f"robustfov completed, using {t1w_to_flirt_actual.name} for FLIRT to AC-PC.")
             # Rename the output_transform_mat_path if robustfov was used, to match the actual FLIRT source
            intended_mat_path_name_base = t1w_to_flirt_actual.name.replace('.nii.gz','').replace('.nii','') + "_to_acpc.mat"
            final_output_transform_mat_path = output_transform_mat_path.parent / intended_mat_path_name_base
            if output_transform_mat_path != final_output_transform_mat_path and output_transform_mat_path.exists():
                 logger.warning(f"Initial .mat path {output_transform_mat_path} might be based on original T1w name. "
                                f"Adjusting based on robustfov output if FLIRT uses it.")
                 # No actual rename here, FLIRT command below will use the correct name from t1w_to_flirt_actual
        else:
            logger.warning(f"robustfov output '{robustfov_out_path.name}' not created or empty. "
                           f"Using original T1w '{input_t1w_path.name}' for FLIRT to AC-PC.")
            final_output_transform_mat_path = output_transform_mat_path # Use the originally passed name
    except Exception as e_fov:
        logger.warning(f"robustfov failed: {e_fov}. Using original T1w '{input_t1w_path.name}' for FLIRT to AC-PC.", exc_info=verbose)
        final_output_transform_mat_path = output_transform_mat_path

    # Ensure the .mat file path for FLIRT output corresponds to t1w_to_flirt_actual
    flirt_output_mat_path = final_output_transform_mat_path.parent / f"{t1w_to_flirt_actual.name.replace('.nii.gz','').replace('.nii','')}_to_acpc.mat"


    flirt_cmd = [
        "flirt", "-in", str(t1w_to_flirt_actual), "-ref", str(mni_template_path),
        "-out", str(output_acpc_t1w_path), "-omat", str(flirt_output_mat_path), # Use dynamically determined mat path
        "-dof", "6", "-cost", "corratio",
        "-searchrx", "-90", "90", "-searchry", "-90", "90", "-searchrz", "-90", "90",
        "-interp", "trilinear"
    ]
    try:
        logger.debug(f"Running FLIRT command: {' '.join(flirt_cmd)}")
        run_cmd(flirt_cmd, verbose=verbose)
        if output_acpc_t1w_path.exists() and output_acpc_t1w_path.stat().st_size > 0 and \
           flirt_output_mat_path.exists() and flirt_output_mat_path.stat().st_size > 0 :
            logger.info(f"Successfully created AC-PC aligned T1w: {output_acpc_t1w_path.name} and transform: {flirt_output_mat_path.name}")
            # If the originally passed mat path was different due to robustfov, try to rename FLIRT's output to it, or just inform user.
            # For simplicity, we ensure `output_transform_mat_path` in the calling function (`main_wf`) is updated based on `t1w_to_flirt_actual`.
            # The `output_transform_mat_path` argument to this function might be slightly misleading if robustfov changes the source.
            # The critical return is t1w_to_flirt_actual, so the caller can construct the correct .mat name.
            return True, t1w_to_flirt_actual
        else:
            logger.error(f"FLIRT ran but output AC-PC aligned T1w or transform ({flirt_output_mat_path.name}) not created/empty.")
            return False, t1w_to_flirt_actual
    except Exception as e:
        logger.error(f"AC-PC alignment using FLIRT failed: {e}", exc_info=verbose)
        return False, t1w_to_flirt_actual

# ... (rest of the utility functions: invert_fsl_transform, resample_volume_to_native_space, etc. remain unchanged) ...
def invert_fsl_transform(
    input_mat_path: Path,
    output_inverse_mat_path: Path,
    logger: logging.Logger,
    verbose: bool = False
) -> bool:
    logger.info(f"Inverting transform {input_mat_path.name} -> {output_inverse_mat_path.name}")
    require_cmds(["convert_xfm"], logger=logger)
    cmd = ["convert_xfm", "-inverse", str(input_mat_path), "-omat", str(output_inverse_mat_path)]
    try:
        run_cmd(cmd, verbose=verbose)
        if output_inverse_mat_path.exists() and output_inverse_mat_path.stat().st_size > 0:
            logger.info(f"Successfully inverted transform: {output_inverse_mat_path.name}")
            return True
        else:
            logger.error(f"convert_xfm ran but output inverse transform not created or empty.")
            return False
    except Exception as e:
        logger.error(f"Transform inversion failed: {e}", exc_info=verbose)
        return False

def resample_volume_to_native_space(
    input_acpc_slab_volume_path: Path,
    output_native_slab_volume_path: Path,
    original_native_t1w_ref_path: Path, 
    acpc_to_native_t1w_mat_path: Path, 
    logger: logging.Logger,
    interpolation: str = "trilinear",
    verbose: bool = False
) -> bool:
    logger.info(f"Resampling AC-PC slab {input_acpc_slab_volume_path.name} to native space -> {output_native_slab_volume_path.name}")
    require_cmds(["flirt"], logger=logger)

    try:
        in_img = nib.load(str(input_acpc_slab_volume_path))
        ref_img = nib.load(str(original_native_t1w_ref_path))
        logger.debug(f"  Input AC-PC slab ({input_acpc_slab_volume_path.name}) affine:\n{in_img.affine}")
        logger.debug(f"  Reference Native T1w ({original_native_t1w_ref_path.name}) affine:\n{ref_img.affine}")
        logger.debug(f"  Using FSL matrix {acpc_to_native_t1w_mat_path.name} for ACPC->Native resampling.")
    except Exception as e_log_aff:
        logger.warning(f"Could not log affines for resampling debug: {e_log_aff}")

    cmd = [
        "flirt",
        "-in", str(input_acpc_slab_volume_path),    
        "-ref", str(original_native_t1w_ref_path), 
        "-applyxfm",
        "-init", str(acpc_to_native_t1w_mat_path),  
        "-out", str(output_native_slab_volume_path),
        "-interp", interpolation
    ]
    try:
        logger.debug(f"Running FLIRT for resampling to native: {' '.join(cmd)}")
        run_cmd(cmd, verbose=verbose)
        if output_native_slab_volume_path.exists() and output_native_slab_volume_path.stat().st_size > 0:
            out_img = nib.load(str(output_native_slab_volume_path))
            ref_img_loaded = nib.load(str(original_native_t1w_ref_path))
            if not np.allclose(out_img.affine, ref_img_loaded.affine):
                logger.warning(f"Output affine for {output_native_slab_volume_path.name} does not match reference T1w affine!")
                logger.debug(f"Output affine:\n{out_img.affine}")
                logger.debug(f"Reference affine:\n{ref_img_loaded.affine}")
            logger.info(f"Successfully resampled to native space: {output_native_slab_volume_path.name}")
            return True
        else:
            logger.error(f"FLIRT resampling ran but output native volume not created or empty for {output_native_slab_volume_path.name}.")
            return False
    except Exception as e:
        logger.error(f"Resampling to native space failed for {input_acpc_slab_volume_path.name}: {e}", exc_info=verbose)
        return False

def get_volume_bounding_box_voxel_coords(volume_path: Path, logger: logging.Logger) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        img = nib.load(str(volume_path))
        data = img.get_fdata().astype(bool) 
        if not np.any(data):
            logger.debug(f"Volume {volume_path.name} is empty, no bounding box.")
            return None
        coords = np.argwhere(data)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        return min_coords, max_coords
    except Exception as e:
        logger.error(f"Failed to get bounding box for {volume_path.name}: {e}", exc_info=True)
        return None

def crop_nifti_volume_fslroi(
    input_volume_path: Path, output_cropped_path: Path,
    min_coords_xyz: np.ndarray, max_coords_xyz: np.ndarray,
    logger: logging.Logger, verbose: bool = False
) -> bool:
    logger.info(f"Cropping {input_volume_path.name} to {output_cropped_path.name} using fslroi.")
    x_min, y_min, z_min = min_coords_xyz.astype(int)
    x_max, y_max, z_max = max_coords_xyz.astype(int)

    x_size = x_max - x_min + 1
    y_size = y_max - y_min + 1
    z_size = z_max - z_min + 1

    if x_size <= 0 or y_size <= 0 or z_size <= 0:
        logger.error(f"Invalid crop dimensions for {input_volume_path.name}: sizes must be positive. "
                     f"Calculated sizes: x={x_size}, y={y_size}, z={z_size} from min/max {min_coords_xyz}/{max_coords_xyz}")
        return False

    fslroi_cmd = [
        "fslroi", str(input_volume_path), str(output_cropped_path),
        str(x_min), str(x_size), str(y_min), str(y_size), str(z_min), str(z_size)
    ]
    try:
        logger.debug(f"Running fslroi command for cropping: {' '.join(fslroi_cmd)}")
        run_cmd(fslroi_cmd, verbose=verbose)
        if output_cropped_path.exists() and output_cropped_path.stat().st_size > 0:
            logger.info(f"Successfully cropped volume: {output_cropped_path.name}")
            return True
        else:
            logger.error(f"fslroi ran for crop but output not created or empty: {output_cropped_path.name}")
            return False
    except Exception as e:
        logger.error(f"Cropping with fslroi failed for {input_volume_path.name}: {e}", exc_info=verbose)
        return False

def extract_volumetric_slab_fslroi(
    input_cropped_volume_path: Path, output_slab_path: Path, slab_label: str,
    orientation_axis_idx: int, slab_start_voxel_coord_in_cropped_vol: int,
    slab_thickness_voxels: int, cropped_vol_dims_xyz: Tuple[int, int, int],
    logger: logging.Logger, verbose: bool = False
) -> bool:
    logger.info(f"Extracting {slab_label} from {input_cropped_volume_path.name} -> {output_slab_path.name}")
    nx_cropped, ny_cropped, nz_cropped = cropped_vol_dims_xyz

    x_min, y_min, z_min = 0, 0, 0
    x_size, y_size, z_size = nx_cropped, ny_cropped, nz_cropped

    if orientation_axis_idx == 0: 
        x_min = slab_start_voxel_coord_in_cropped_vol
        x_size = min(slab_thickness_voxels, nx_cropped - x_min)
    elif orientation_axis_idx == 1: 
        y_min = slab_start_voxel_coord_in_cropped_vol
        y_size = min(slab_thickness_voxels, ny_cropped - y_min)
    elif orientation_axis_idx == 2: 
        z_min = slab_start_voxel_coord_in_cropped_vol
        z_size = min(slab_thickness_voxels, nz_cropped - z_min)
    else:
        logger.error(f"Invalid orientation_axis_idx for slab extraction: {orientation_axis_idx}"); return False

    if x_size <= 0 or y_size <= 0 or z_size <= 0:
        logger.error(f"Invalid slab dimensions for {slab_label} of {input_cropped_volume_path.name}: sizes must be positive. "
                     f"Calculated: x_start={int(x_min)}, x_size={int(x_size)}; "
                     f"y_start={int(y_min)}, y_size={int(y_size)}; "
                     f"z_start={int(z_min)}, z_size={int(z_size)}"); return False

    fslroi_cmd = [
        "fslroi", str(input_cropped_volume_path), str(output_slab_path),
        str(int(x_min)), str(int(x_size)), str(int(y_min)), str(int(y_size)), str(int(z_min)), str(int(z_size))
    ]
    try:
        logger.debug(f"Running fslroi for slab extraction: {' '.join(fslroi_cmd)}")
        run_cmd(fslroi_cmd, verbose=verbose)
        if output_slab_path.exists() and output_slab_path.stat().st_size > 0:
            logger.info(f"Successfully extracted {slab_label}: {output_slab_path.name}"); return True
        else:
            logger.error(f"fslroi ran for {slab_label} but output not created or empty: {output_slab_path.name}"); return False
    except Exception as e:
        logger.error(f"Slab extraction with fslroi failed for {slab_label} of {input_cropped_volume_path.name}: {e}", exc_info=verbose); return False

def vol_to_mesh(volume_path: Path, output_mesh_path: Path, no_smooth: bool, logger: logging.Logger) -> bool:
    logger.info(f"Attempting to convert volume {volume_path.name} to mesh {output_mesh_path.name}")
    try:
        img_check = nib.load(str(volume_path))
        data_check = img_check.get_fdata()
        is_effectively_empty = False
        if data_check.dtype == np.uint8 or data_check.dtype == bool: 
            if data_check.size > 0 and np.sum(data_check) < 10: 
                is_effectively_empty = True
        elif data_check.size > 0 and np.max(np.abs(data_check)) < 0.01: 
             is_effectively_empty = True
        
        if is_effectively_empty:
            logger.info(f"Volume {volume_path.name} appears effectively empty (sum/max criteria). Skipping mesh generation.")
            return False 
    except Exception as e:
        logger.error(f"Could not load volume {volume_path.name} to check if empty: {e}", exc_info=True)
        return False 

    temp_gii_path = output_mesh_path.with_name(f"{output_mesh_path.stem}_{uuid.uuid4().hex[:6]}.surf.gii")
    mesh: Optional[trimesh.Trimesh] = None

    try:
        volume_to_gifti(str(volume_path), str(temp_gii_path), level=0.5) 
        if temp_gii_path.exists() and temp_gii_path.stat().st_size > 0:
            mesh = gifti_to_trimesh(str(temp_gii_path)) 
        else:
            logger.warning(f"GIFTI file {temp_gii_path.name} was not created or is empty from {volume_path.name}.")
    except Exception as e_vtg:
        logger.error(f"volume_to_gifti or gifti_to_trimesh failed for {volume_path.name}: {e_vtg}", exc_info=True)
    finally:
        if temp_gii_path.exists(): temp_gii_path.unlink(missing_ok=True)

    if not isinstance(mesh, trimesh.Trimesh) or (hasattr(mesh, 'is_empty') and mesh.is_empty):
        logger.warning(f"Mesh from {volume_path.name} is not a valid Trimesh object or is empty after initial conversion. Type: {type(mesh)}")
        return False 
    
    try:
        logger.debug(f"Initial mesh for {output_mesh_path.name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces. Watertight: {mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'}")
        mesh.process(validate=True) 
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
             logger.error(f"Mesh became invalid or empty after mesh.process() for {output_mesh_path.name}! Reverting to pre-process state if possible or failing.")
             return False 
        logger.debug(f"After mesh.process for {output_mesh_path.name}, Watertight: {mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'}")

        if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
            logger.debug(f"Attempting to fill holes for non-watertight mesh {output_mesh_path.name}...")
            trimesh.repair.fill_holes(mesh)
            logger.info(f"After fill_holes for {output_mesh_path.name}, Watertight: {mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'}")
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                 logger.error(f"Mesh became invalid or empty after fill_holes for {output_mesh_path.name}!")
                 return False

        if not no_smooth:
            logger.info(f"Attempting Taubin smoothing for {output_mesh_path.name}...")
            if mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0: 
                smoothed_mesh_candidate = trimesh.smoothing.filter_taubin(mesh, iterations=10, lamb=0.5, nu=-0.53)
                if isinstance(smoothed_mesh_candidate, trimesh.Trimesh) and not smoothed_mesh_candidate.is_empty:
                    mesh = smoothed_mesh_candidate
                    logger.debug(f"Successfully applied Taubin smoothing to {output_mesh_path.name}.")
                else:
                    logger.warning(f"Taubin smoothing for {output_mesh_path.name} returned invalid/empty. Using pre-smoothing state.")
            else:
                logger.warning(f"Skipping Taubin smoothing for {output_mesh_path.name} as mesh is invalid/empty pre-smoothing.")
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                logger.error(f"Mesh became invalid or empty after smoothing block for {output_mesh_path.name}!")
                return False
        logger.debug(f"After smoothing block for {output_mesh_path.name}, type(mesh) is {type(mesh)}")
        
        logger.debug(f"Attempting to fix normals for {output_mesh_path.name}...")
        mesh.fix_normals(multibody=True)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            logger.error(f"Mesh became invalid or empty after fix_normals for {output_mesh_path.name}!")
            return False
        logger.debug(f"After fix_normals for {output_mesh_path.name}, type(mesh) is {type(mesh)}")
        
        is_trimesh = isinstance(mesh, trimesh.Trimesh)
        watertight_status = mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'
        logger.debug(f"Preparing to export {output_mesh_path.name}. Mesh final check - Type: {type(mesh)}, Is Trimesh: {is_trimesh}, Watertight: {watertight_status}")

        if not is_trimesh:
             logger.error(f"Mesh variable is NOT a Trimesh object before export for {output_mesh_path.name}. Type: {type(mesh)}")
             return False

        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_mesh_path))
        if output_mesh_path.exists() and output_mesh_path.stat().st_size > 0:
             logger.info(f"Successfully exported mesh: {output_mesh_path}")
             return True
        else:
             logger.error(f"Mesh export for {output_mesh_path.name} failed or file is empty.")
             return False

    except AttributeError as e_attrib_export:
        logger.error(f"AttributeError during processing or export of {output_mesh_path.name}: {e_attrib_export}. Mesh type: {type(mesh)}", exc_info=True)
        return False
    except Exception as e_export:
        logger.error(f"Failed to process or export mesh {output_mesh_path.name}: {e_export}", exc_info=True)
        return False

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-material brain slab components for 3D printing.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory.")
    parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-01).")
    parser.add_argument("--t1w_input", required=False, default=None,
                        help="Path to the T1w NIfTI image to be used for AC-PC alignment. "
                             "If None, attempts to find 'desc-preproc_T1w.nii.gz' in subject's anat dir.")
    parser.add_argument("--mni_template", type=str, default=str(DEFAULT_MNI_TEMPLATE_PATH) if DEFAULT_MNI_TEMPLATE_PATH else None,
                        help="Path to MNI152 T1 1mm brain template for AC-PC alignment. "
                             f"Default tries FSLDIR ({MNI_TEMPLATE_NAME_DEFAULT}).")
    parser.add_argument("--output_dir", default="./multi_material_slabs",
                        help="Base output directory for final STL components and intermediate slab volumes.")
    parser.add_argument("--work_dir", default=None,
                        help="Directory for all intermediate files. If None, a temporary one is created under output_dir.")
    parser.add_argument("--session", default=None, help="BIDS session entity (e.g., ses-01).")
    parser.add_argument("--run", default=None, help="BIDS run entity (e.g., run-1).")
    parser.add_argument("--slab_thickness", type=float, default=5.0, help="Thickness of each slab in mm.")
    parser.add_argument("--slab_orientation", choices=["axial", "coronal", "sagittal"], default="axial",
                        help="Orientation for slicing (relative to AC-PC aligned space).")
    parser.add_argument("--brain_mask_inflate_mm", type=float, default=1.0,
                        help="Amount (in mm) to inflate the brain mask before using it.")
    parser.add_argument("--voxel_resolution", type=float, default=0.5,
                        help="Voxel size (mm) for the high-resolution AC-PC master grid.")
    parser.add_argument("--pv_threshold", type=float, default=0.5,
                        help="Threshold for binarizing partial volume images from mesh2voxel.")
    parser.add_argument("--no_final_mesh_smoothing", action="store_true",
                        help="Disable Taubin smoothing on the final component meshes (STLs).")
    parser.add_argument("--skip_outer_csf", action="store_true", help="Skip generation of the outer CSF component.")
    parser.add_argument("--no_clean", action="store_true", help="Keep work directory if it was temporary.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")
    parser.add_argument("--flatten_caps_tolerance_mm", type=float, default=0.5,
                        help="EXPERIMENTAL: If set (e.g. 0.5), attempts to flatten mesh caps by snapping vertices. "
                             "Value is the tolerance in mm for selecting cap vertices in AC-PC space. "
                             "Requires fslpy for accurate transformations.")
    parser.add_argument("--output_final_slabs_in_acpc_space", action="store_true",
                        help="If set, the final output slab meshes will be in ACPC space. "
                             "Otherwise, they are in native T1w space (default).")
    return parser


def main_wf(args: argparse.Namespace, L: logging.Logger, work_dir: Path, stl_output_base_dir: Path, runlog: Dict[str, Any]):
    runlog["steps"].append("Starting multi-material slab component generation.")
    L.info(f"Target AC-PC aligned voxel resolution: {args.voxel_resolution} mm")
    L.info(f"Slab thickness: {args.slab_thickness}mm, Orientation (AC-PC): {args.slab_orientation}")
    L.info(f"BrainMask inflation: {args.brain_mask_inflate_mm}mm")
    if args.output_final_slabs_in_acpc_space:
        L.info("Final slab meshes will be output in ACPC space.")
    else:
        L.info("Final slab meshes will be output in Native T1w space.")


    mrtrix_cmds = ["mrgrid", "mesh2voxel"]
    fsl_cmds = ["flirt", "fslroi", "robustfov", "convert_xfm"]
    require_cmds(mrtrix_cmds + fsl_cmds, logger=L)

    # Define directories
    acpc_align_dir = work_dir / "00_acpc_alignment"
    parent_surf_gen_dir = work_dir / "01_parent_surface_generation"
    full_vol_voxelized_dir = work_dir / "02_full_acpc_voxelized_pvs"
    full_vol_binarized_dir = work_dir / "03_full_acpc_binarized_masks"
    cropped_full_vol_dir = work_dir / "04_cropped_full_acpc_masks"
    volumetric_slabs_acpc_dir = work_dir / "05_volumetric_slabs_acpc_nifti"
    material_slabs_acpc_vol_dir = work_dir / "06_material_slabs_acpc_volumes_nifti"
    material_slabs_native_vol_dir = work_dir / "07_material_slabs_native_volumes_nifti" # Always created for native meshing step
    
    dirs_to_create = [acpc_align_dir, parent_surf_gen_dir, full_vol_voxelized_dir,
                   full_vol_binarized_dir, cropped_full_vol_dir, volumetric_slabs_acpc_dir,
                   material_slabs_acpc_vol_dir, material_slabs_native_vol_dir]
        
    for d_path in dirs_to_create:
        d_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: AC-PC Alignment of Input T1w ---
    L.info("--- Step 1: AC-PC Alignment of Input T1w ---")
    original_t1w_path_for_ref: Optional[Path] = None
    if args.t1w_input:
        original_t1w_path_for_ref = Path(args.t1w_input)
        if not original_t1w_path_for_ref.exists():
            L.error(f"Specified T1w input not found: {original_t1w_path_for_ref}"); return False
    else:
        try:
            original_t1w_path_for_ref = Path(flexible_match(
                base_dir=(Path(args.subjects_dir) / args.subject_id / "anat"),
                subject_id=args.subject_id, descriptor="preproc", suffix="T1w", ext=".nii.gz",
                session=args.session, run=args.run, logger=L ))
        except FileNotFoundError:
            L.error(f"Default T1w preproc image not found for {args.subject_id}."); return False

    t1w_acpc_aligned_path = acpc_align_dir / f"{args.subject_id}_T1w_acpc.nii.gz"
    mni_template_for_acpc = Path(args.mni_template)
    
    # Define the initial name for the .mat file based on the *original* T1w.
    # acpc_align_t1w will handle the actual FLIRT source and ensure the output .mat file
    # from FLIRT is named according to the actual FLIRT input (e.g. robustfov output)
    # but we need a placeholder path to pass, and then we'll get the *actual* one used.
    initial_mat_path_guess = acpc_align_dir / f"{original_t1w_path_for_ref.name.replace('.nii.gz','').replace('.nii','')}_to_acpc.mat"

    acpc_success, actual_flirt_src_path = acpc_align_t1w(
        original_t1w_path_for_ref,
        t1w_acpc_aligned_path,
        initial_mat_path_guess, # Pass the initial guess, acpc_align_t1w will ensure FLIRT saves with correct name
        mni_template_for_acpc,
        acpc_align_dir, L, args.verbose
    )
    if not acpc_success or actual_flirt_src_path is None:
        L.error("AC-PC alignment step failed."); return False
    
    # The actual .mat file is named based on actual_flirt_src_path
    t1w_to_acpc_fsl_mat_path = acpc_align_dir / f"{actual_flirt_src_path.name.replace('.nii.gz','').replace('.nii','')}_to_acpc.mat"
    if not t1w_to_acpc_fsl_mat_path.exists(): # Double check it was created by FLIRT inside acpc_align_t1w
        L.error(f"FLIRT output .mat file {t1w_to_acpc_fsl_mat_path} not found after acpc_align_t1w call. Aborting."); return False

    runlog["steps"].append(f"AC-PC T1w: {t1w_acpc_aligned_path.name}, Native-to-ACPC FSL XFM: {t1w_to_acpc_fsl_mat_path.name}")

    acpc_to_native_fsl_mat_path = acpc_align_dir / f"{actual_flirt_src_path.name.replace('.nii.gz','').replace('.nii','')}_acpc_to_native.mat"
    if not invert_fsl_transform(t1w_to_acpc_fsl_mat_path, acpc_to_native_fsl_mat_path, L, args.verbose):
        L.error("Failed to invert AC-PC alignment FSL transform."); return False
    runlog["steps"].append(f"Inverse FSL XFM (ACPC->Native): {acpc_to_native_fsl_mat_path.name}")

    # --- Derive World-to-World Affines ---
    native_to_acpc_world_affine: Optional[np.ndarray] = None
    acpc_to_native_world_affine: Optional[np.ndarray] = None

    if FSLPY_AVAILABLE: # These are crucial for cap flattening
        L.info("Attempting to derive world-to-world affines using fslpy...")
        native_to_acpc_world_affine = get_flirt_world_to_world_affine(
            fsl_flirt_mat_path=t1w_to_acpc_fsl_mat_path,
            src_image_path=actual_flirt_src_path, # This is the *actual* source used in FLIRT (e.g. robustfov output)
            ref_image_path=mni_template_for_acpc,
            logger=L
        )
        if native_to_acpc_world_affine is not None:
            L.info("Successfully derived native-to-ACPC world affine. Calculating its inverse.")
            try:
                acpc_to_native_world_affine = np.linalg.inv(native_to_acpc_world_affine)
                L.info("Successfully calculated ACPC-to-native world affine (inverse).")
            except np.linalg.LinAlgError as e_inv:
                L.error(f"Failed to invert native-to-ACPC world affine: {e_inv}. "
                        "This will affect cap flattening if native output is chosen.", exc_info=True)
                acpc_to_native_world_affine = None # Critical for native output path of flattening
        else:
            L.error("Failed to derive native-to-ACPC world affine using fslpy. Cap flattening will be impacted.")
            acpc_to_native_world_affine = None 
    else: 
        L.warning("fslpy is not available. World-to-world affines cannot be reliably computed. "
                  "Step 4 (voxelization) may be misaligned. Cap flattening will be disabled if it relies on these.")
        runlog["warnings"].append("fslpy not available. Critical affine transforms skipped. Potential misalignment and cap flattening disabled.")
    
    # Disable cap flattening if key affines are missing
    if args.flatten_caps_tolerance_mm is not None:
        if native_to_acpc_world_affine is None:
            L.warning("Disabling cap flattening as native_to_acpc_world_affine could not be computed.")
            args.flatten_caps_tolerance_mm = None
        elif not args.output_final_slabs_in_acpc_space and acpc_to_native_world_affine is None:
            # If native output is desired for flattened meshes, we need acpc_to_native_world_affine
            L.warning("Disabling cap flattening for native output as acpc_to_native_world_affine could not be computed.")
            args.flatten_caps_tolerance_mm = None


    # --- Step 2: Creating High-Resolution Master AC-PC Template ---
    # ... (This section remains unchanged) ...
    L.info("--- Step 2: Creating High-Resolution Master AC-PC Template ---")
    master_hires_acpc_template_path = acpc_align_dir / f"{args.subject_id}_master_hires_acpc_template_{args.voxel_resolution}mm.nii.gz"
    if not regrid_to_resolution(t1w_acpc_aligned_path, master_hires_acpc_template_path, args.voxel_resolution, L, args.verbose):
        L.error("Failed to create high-resolution master AC-PC template."); return False
    runlog["steps"].append(f"High-res master AC-PC template created: {master_hires_acpc_template_path.name}")
    try:
        master_template_nifti_image_obj = nib.load(str(master_hires_acpc_template_path))
        master_template_hires_voxel_sizes = np.array(master_template_nifti_image_obj.header.get_zooms()[:3])
    except Exception as e_load_master_template:
        L.error(f"Failed to load master AC-PC template {master_hires_acpc_template_path}: {e_load_master_template}", exc_info=True); return False

    # --- Step 3: Generating full parent T1-space surfaces ---
    # ... (This section remains unchanged) ...
    L.info("--- Step 3: Generating full parent T1-space surfaces (Trimesh objects) ---")
    parent_meshes_trimesh: Dict[str, Optional[trimesh.Trimesh]] = {}
    L.info(f"Generating BrainMask surface with {args.brain_mask_inflate_mm}mm inflation...")
    bm_mesh = generate_single_brain_mask_surface(
        args.subjects_dir, args.subject_id, "T1",
        inflate_mm=args.brain_mask_inflate_mm, no_smooth=False,
        run=args.run, session=args.session,
        tmp_dir=parent_surf_gen_dir / "bm_work", logger=L, verbose=args.verbose)
    if bm_mesh and not bm_mesh.is_empty: parent_meshes_trimesh["BrainMask"] = bm_mesh
    else: L.warning("BrainMask surface generation failed or resulted in an empty mesh.")
    L.info("Generating Pial Complex components...")
    pial_cortical_for_complex, pial_other_for_complex_from_preset, _ = parse_preset("pial_brain")
    pial_other_explicit_for_pial_complex = set()
    for item in pial_other_for_complex_from_preset:
        if item == "cerebellum": pial_other_explicit_for_pial_complex.add("cerebellum")
        else: pial_other_explicit_for_pial_complex.add(item)
    pial_comp_meshes = generate_brain_surfaces(
        args.subjects_dir, args.subject_id, "T1",
        tuple(pial_cortical_for_complex), list(pial_other_explicit_for_pial_complex),
        [], [], args.run, args.session, args.verbose, str(parent_surf_gen_dir / "pial_complex_work"))
    for k, v_mesh in pial_comp_meshes.items():
        if v_mesh and not v_mesh.is_empty: parent_meshes_trimesh[k] = v_mesh
    L.info("Generating White Complex components...")
    white_cortical_for_complex, white_other_for_complex_from_preset, _ = parse_preset("white_brain")
    white_other_explicit_for_white_complex = set()
    for item in white_other_for_complex_from_preset:
        if item == "cerebellum_wm": white_other_explicit_for_white_complex.add("cerebellum_wm")
        elif item == "cerebellum":
            white_other_explicit_for_white_complex.add("cerebellum_wm") 
        else: white_other_explicit_for_white_complex.add(item)

    white_comp_meshes = generate_brain_surfaces(
        args.subjects_dir, args.subject_id, "T1",
        tuple(white_cortical_for_complex), list(white_other_explicit_for_white_complex),
        [], [], args.run, args.session, args.verbose, str(parent_surf_gen_dir / "white_complex_work"))
    for k, v_mesh in white_comp_meshes.items():
        if v_mesh and not v_mesh.is_empty: parent_meshes_trimesh[k] = v_mesh

    if is_vtk_available(): 
        fs_input_dir = Path(args.subjects_dir) / "sourcedata" / "freesurfer" / args.subject_id
        if fs_input_dir.is_dir():
            five_tt_gen_dir = parent_surf_gen_dir / "5ttgen_work_multimat"
            five_tt_gen_dir.mkdir(parents=True, exist_ok=True)
            if run_5ttgen_hsvs_save_temp_bids(args.subject_id, str(fs_input_dir), str(five_tt_gen_dir), args.session, verbose=args.verbose):
                vtk_meshes = load_subcortical_and_ventricle_meshes(str(five_tt_gen_dir)) 
                for k, v_mesh in vtk_meshes.items():
                    if v_mesh and not v_mesh.is_empty: parent_meshes_trimesh[k] = v_mesh
            else: L.warning("5ttgen command failed, SGM/Ventricles might be missing or incomplete.")
        else: L.warning(f"FreeSurfer dir for 5ttgen not found: {fs_input_dir}, skipping SGM/Ventricles.")
    else: L.warning("VTK/PyVista not available, skipping SGM/Ventricle generation from 5ttgen VTK files.")

    valid_parent_meshes_trimesh = {k:v for k,v in parent_meshes_trimesh.items() if v and not v.is_empty}
    if not valid_parent_meshes_trimesh: L.error("No valid parent Trimesh surfaces generated. Aborting."); return False
    L.info(f"Generated {len(valid_parent_meshes_trimesh)} initial parent Trimesh surfaces: {list(valid_parent_meshes_trimesh.keys())}")

    # --- Step 4: Voxelizing full parent surfaces onto Master AC-PC Template ---
    # ... (This section remains unchanged; uses native_to_acpc_world_affine if available) ...
    L.info(f"--- Step 4: Voxelizing full parent surfaces onto Master AC-PC Template ({master_hires_acpc_template_path.name}) ---")
    full_acpc_binarized_masks: Dict[str, Path] = {}
    temp_obj_export_dir_for_acpc_voxelization = full_vol_voxelized_dir / "temp_acpc_aligned_objs"
    temp_obj_export_dir_for_acpc_voxelization.mkdir(parents=True, exist_ok=True)

    for parent_name, trimesh_native_obj in valid_parent_meshes_trimesh.items():
        if trimesh_native_obj is None: continue 

        safe_name = parent_name.replace("_", "-").replace(" ", "-")
        temp_acpc_aligned_surf_obj_path = temp_obj_export_dir_for_acpc_voxelization / f"{args.subject_id}_desc-{safe_name}_full_surf_acpc-aligned.obj"
        mesh_to_voxelize_acpc = trimesh_native_obj.copy()

        if native_to_acpc_world_affine is not None:
            L.debug(f"Transforming native surface '{parent_name}' to AC-PC space before voxelization.")
            mesh_to_voxelize_acpc.apply_transform(native_to_acpc_world_affine)
        else:
            L.critical(f"CRITICAL: Native-to-ACPC world affine is MISSING for surface '{parent_name}'. "
                      f"Voxelization onto AC-PC template ('{master_hires_acpc_template_path.name}') will be INCORRECT.")
            runlog["warnings"].append(f"CRITICAL_VOXELIZATION_ERROR: {parent_name} voxelized without AC-PC transform.")
            
        try:
            mesh_to_voxelize_acpc.export(str(temp_acpc_aligned_surf_obj_path))
        except Exception as e:
            L.error(f"Failed to export AC-PC aligned surface {parent_name} to OBJ '{temp_acpc_aligned_surf_obj_path}': {e}", exc_info=True)
            continue

        full_pv_path = full_vol_voxelized_dir / f"{args.subject_id}_desc-{safe_name}_full_acpc_pv.nii.gz"
        if not mesh_to_partial_volume(temp_acpc_aligned_surf_obj_path, master_hires_acpc_template_path, full_pv_path, L, args.verbose):
            L.warning(f"Failed to voxelize AC-PC aligned {parent_name}."); continue

        full_bin_path = full_vol_binarized_dir / f"{args.subject_id}_desc-{safe_name}_full_acpc_mask.nii.gz"
        if binarize_volume_file(full_pv_path, full_bin_path, args.pv_threshold, L):
            full_acpc_binarized_masks[parent_name] = full_bin_path
            L.info(f"Generated AC-PC binarized mask: {full_bin_path}. ")
        else: L.warning(f"Failed to binarize full PV for {parent_name}.")
        
        if not args.no_clean and temp_acpc_aligned_surf_obj_path.exists():
             try: temp_acpc_aligned_surf_obj_path.unlink()
             except OSError: L.debug(f"Could not unlink {temp_acpc_aligned_surf_obj_path.name}, possibly already removed.")
    
    if not args.no_clean and temp_obj_export_dir_for_acpc_voxelization.exists():
        try: shutil.rmtree(temp_obj_export_dir_for_acpc_voxelization)
        except OSError: L.debug(f"Could not rmtree {temp_obj_export_dir_for_acpc_voxelization.name}, possibly already removed.")

    if not full_acpc_binarized_masks: L.error("No full surfaces voxelized/binarized. Aborting."); return False
    L.info(f"Voxelized and binarized {len(full_acpc_binarized_masks)} full parent surfaces onto AC-PC grid.")

    # --- Step 5: Determining global bounding box for cropping ---
    # ... (This section remains unchanged) ...
    L.info("--- Step 5: Determining global bounding box for cropping full AC-PC volumes ---")
    overall_min_coords_vox: Optional[np.ndarray] = None
    overall_max_coords_vox: Optional[np.ndarray] = None
    for struct_name, mask_path in full_acpc_binarized_masks.items():
        bounds_vox = get_volume_bounding_box_voxel_coords(mask_path, L)
        if bounds_vox:
            min_v, max_v = bounds_vox
            if overall_min_coords_vox is None: 
                overall_min_coords_vox, overall_max_coords_vox = min_v.copy(), max_v.copy()
            else:
                overall_min_coords_vox = np.minimum(overall_min_coords_vox, min_v)
                overall_max_coords_vox = np.maximum(overall_max_coords_vox, max_v)
    
    if overall_min_coords_vox is None or overall_max_coords_vox is None:
        L.error("Could not determine global bounding box for cropping. Aborting."); return False
    L.info(f"Global voxel bounding box for cropping (min/max xyz, inclusive): {overall_min_coords_vox} / {overall_max_coords_vox}")

    cropped_acpc_binarized_masks: Dict[str, Path] = {}
    cropped_acpc_grid_affine_for_caps: Optional[np.ndarray] = None 
    
    for struct_name, full_mask_path in full_acpc_binarized_masks.items():
        safe_name = struct_name.replace("_", "-").replace(" ", "-") 
        cropped_path = cropped_full_vol_dir / f"{args.subject_id}_desc-{safe_name}_full_acpc_mask_cropped.nii.gz"
        if crop_nifti_volume_fslroi(full_mask_path, cropped_path, overall_min_coords_vox, overall_max_coords_vox, L, args.verbose):
            cropped_acpc_binarized_masks[struct_name] = cropped_path
            if cropped_acpc_grid_affine_for_caps is None: 
                try:
                    cropped_img_for_affine = nib.load(str(cropped_path))
                    cropped_acpc_grid_affine_for_caps = cropped_img_for_affine.affine
                    L.info(f"Using affine from {cropped_path.name} for ideal cap coordinate calculations.")
                except Exception as e_aff:
                    L.error(f"Could not load affine from {cropped_path.name}: {e_aff}. Cap flattening may be unreliable.")
                    runlog["warnings"].append(f"Failed to get AC-PC cropped grid affine for cap flattening: {e_aff}")
        else: L.warning(f"Failed to crop {full_mask_path.name}.")

    if not cropped_acpc_binarized_masks: L.error("No volumes successfully cropped. Aborting."); return False
    L.info(f"Cropped {len(cropped_acpc_binarized_masks)} full AC-PC volumes.")

    if args.flatten_caps_tolerance_mm is not None and cropped_acpc_grid_affine_for_caps is None:
        L.error("Cap flattening enabled, but could not determine affine of the cropped AC-PC grid. Disabling feature.")
        runlog["warnings"].append("Failed to get AC-PC cropped grid affine; cap flattening disabled.")
        args.flatten_caps_tolerance_mm = None # Disable if this crucial affine is missing

    # --- Step 6: Performing volumetric slicing (in AC-PC space) ---
    # ... (This section remains unchanged) ...
    L.info("--- Step 6: Performing volumetric slicing (in AC-PC space) ---")
    orientation_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    slice_axis_idx = orientation_map[args.slab_orientation]
    runlog["slice_axis_idx_acpc"] = slice_axis_idx

    example_cropped_vol_path = next(iter(cropped_acpc_binarized_masks.values()), None)
    if not example_cropped_vol_path: L.error("No cropped volumes for slicing. Aborting."); return False
    try:
        example_cropped_img = nib.load(str(example_cropped_vol_path))
        cropped_vol_dims_xyz = np.array(example_cropped_img.shape[:3], dtype=int)
    except Exception as e: L.error(f"Could not load {example_cropped_vol_path} for slicing dims: {e}", exc_info=True); return False

    slicing_axis_voxel_size_mm = master_template_hires_voxel_sizes[slice_axis_idx]
    slab_thickness_voxels = max(1, int(round(args.slab_thickness / slicing_axis_voxel_size_mm)))
    L.info(f"Slicing axis {slice_axis_idx}. Voxel size on this axis: {slicing_axis_voxel_size_mm:.3f}mm. "
           f"Slab thickness: {args.slab_thickness}mm => {slab_thickness_voxels} voxels.")

    num_slabs_generated = 0
    current_slab_start_voxel_in_cropped = 0 
    volumetric_slabs_acpc_struct_files: Dict[int, Dict[str, Path]] = {} 
    slab_definitions_voxel_acpc: List[Dict[str, Any]] = [] 

    total_extent_cropped_vox = cropped_vol_dims_xyz[slice_axis_idx]

    while current_slab_start_voxel_in_cropped < total_extent_cropped_vox:
        slab_idx = num_slabs_generated
        volumetric_slabs_acpc_struct_files[slab_idx] = {}
        
        actual_slab_thickness_this_iteration_vox = min(slab_thickness_voxels, total_extent_cropped_vox - current_slab_start_voxel_in_cropped)
        if actual_slab_thickness_this_iteration_vox <= 0: break 

        slab_definitions_voxel_acpc.append({
            "slab_idx": slab_idx,
            "start_vox": current_slab_start_voxel_in_cropped,
            "thickness_vox": actual_slab_thickness_this_iteration_vox
        })
        runlog["steps"].append(f"Defining AC-PC slab {slab_idx}: start_vox={current_slab_start_voxel_in_cropped}, thickness_vox={actual_slab_thickness_this_iteration_vox}")

        L.info(f"--- >> Volumetrically Slicing AC-PC Slab Index {slab_idx} (voxels {current_slab_start_voxel_in_cropped} to "
               f"{current_slab_start_voxel_in_cropped + actual_slab_thickness_this_iteration_vox -1} on axis {slice_axis_idx} of cropped volume) << ---")
        
        for struct_name, cropped_vol_path in cropped_acpc_binarized_masks.items():
            safe_name = struct_name.replace("_", "-").replace(" ", "-") 
            slab_nifti_path = volumetric_slabs_acpc_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{safe_name}_acpc_volslab.nii.gz"
            
            if extract_volumetric_slab_fslroi(
                cropped_vol_path, slab_nifti_path, f"slab-{slab_idx}_{safe_name}", slice_axis_idx,
                current_slab_start_voxel_in_cropped, actual_slab_thickness_this_iteration_vox,
                tuple(cropped_vol_dims_xyz), L, args.verbose
            ):
                volumetric_slabs_acpc_struct_files[slab_idx][struct_name] = slab_nifti_path
            else: L.warning(f"Failed to extract AC-PC volumetric slab for {struct_name}, slab {slab_idx}.")
        
        current_slab_start_voxel_in_cropped += actual_slab_thickness_this_iteration_vox
        num_slabs_generated += 1

    if num_slabs_generated == 0: L.error("No volumetric AC-PC slabs extracted. Aborting."); return False
    L.info(f"Processed {num_slabs_generated} volumetric AC-PC slabs definitions.")
    runlog["num_slabs_defined"] = num_slabs_generated

    # --- Loop through each defined slab index for material processing ---
    for slab_idx in range(num_slabs_generated):
        L.info(f"--- >>> Processing Materials for Slab {slab_idx + 1}/{num_slabs_generated} <<< ---")
        current_slab_acpc_component_paths = volumetric_slabs_acpc_struct_files.get(slab_idx, {})
        if not current_slab_acpc_component_paths:
            L.warning(f"Slab {slab_idx}: No AC-PC component NIfTI files found for material definition. Skipping slab."); continue

        # --- Step 7: Volumetric Math for Materials (in AC-PC space for this slab) ---
        # ... (This section remains largely unchanged, populates material_acpc_slab_volumes_to_process) ...
        M_KeyStruct_Slab_i_data_dict: Dict[str, Optional[np.ndarray]] = {}
        example_acpc_slab_nifti_path_for_saving_template: Optional[Path] = None

        def load_acpc_slab_nifti(struct_name_key: str) -> Optional[np.ndarray]:
            nonlocal example_acpc_slab_nifti_path_for_saving_template 
            path_to_load = current_slab_acpc_component_paths.get(struct_name_key)
            if path_to_load and path_to_load.exists():
                if example_acpc_slab_nifti_path_for_saving_template is None: 
                    example_acpc_slab_nifti_path_for_saving_template = path_to_load
                return load_nifti_data(path_to_load, L)
            L.debug(f"Slab {slab_idx}: AC-PC NIfTI for '{struct_name_key}' not found at {path_to_load}.")
            return None

        M_KeyStruct_Slab_i_data_dict["BrainMask"] = load_acpc_slab_nifti("BrainMask")
        if M_KeyStruct_Slab_i_data_dict["BrainMask"] is None:
            L.error(f"Slab {slab_idx}: Essential AC-PC BrainMask NIfTI missing or failed to load. Skipping this slab."); continue
        
        zeros_fallback_shape_acpc = M_KeyStruct_Slab_i_data_dict["BrainMask"].shape
        zeros_for_fallback_acpc_slab = np.zeros(zeros_fallback_shape_acpc, dtype=M_KeyStruct_Slab_i_data_dict["BrainMask"].dtype)

        pial_keys = ["pial_L", "pial_R", "corpus_callosum", "cerebellum", "brainstem"] 
        pial_arrs = [load_acpc_slab_nifti(k) for k in pial_keys if k in current_slab_acpc_component_paths]
        pial_union_result = vol_union_numpy([a for a in pial_arrs if a is not None])
        M_KeyStruct_Slab_i_data_dict["PialComplex"] = pial_union_result if pial_union_result is not None else zeros_for_fallback_acpc_slab.copy()

        white_keys = ["white_L", "white_R", "corpus_callosum", "cerebellum_wm", "brainstem"] 
        white_arrs = [load_acpc_slab_nifti(k) for k in white_keys if k in current_slab_acpc_component_paths]
        white_union_result = vol_union_numpy([a for a in white_arrs if a is not None])
        M_KeyStruct_Slab_i_data_dict["WhiteComplex"] = white_union_result if white_union_result is not None else zeros_for_fallback_acpc_slab.copy()

        vent_keys = [k for k in valid_parent_meshes_trimesh if k.startswith("ventricle-")] 
        vent_arrs = [load_acpc_slab_nifti(k) for k in vent_keys if k in current_slab_acpc_component_paths]
        vent_union_result = vol_union_numpy([a for a in vent_arrs if a is not None])
        M_KeyStruct_Slab_i_data_dict["VentriclesCombined"] = vent_union_result if vent_union_result is not None else zeros_for_fallback_acpc_slab.copy()

        sgm_keys = [k for k in valid_parent_meshes_trimesh if k.startswith("subcortical-")] 
        sgm_arrs = [load_acpc_slab_nifti(k) for k in sgm_keys if k in current_slab_acpc_component_paths]
        sgm_union_result = vol_union_numpy([a for a in sgm_arrs if a is not None])
        M_KeyStruct_Slab_i_data_dict["SGMCombined"] = sgm_union_result if sgm_union_result is not None else zeros_for_fallback_acpc_slab.copy()
        
        L.info(f"--- Slab {slab_idx}: Volumetric math for materials (AC-PC space) ---")
        m_bm_acpc, m_pc_acpc, m_wc_acpc, m_vent_acpc, m_sgm_acpc = (
            M_KeyStruct_Slab_i_data_dict["BrainMask"], M_KeyStruct_Slab_i_data_dict["PialComplex"],
            M_KeyStruct_Slab_i_data_dict["WhiteComplex"], M_KeyStruct_Slab_i_data_dict["VentriclesCombined"],
            M_KeyStruct_Slab_i_data_dict["SGMCombined"] )
        
        if example_acpc_slab_nifti_path_for_saving_template is None: 
            L.error(f"Slab {slab_idx}: No example AC-PC NIfTI slab to use as template for saving. Skipping save for this slab's materials."); continue
        try:
            acpc_slab_template_img_for_saving = nib.load(str(example_acpc_slab_nifti_path_for_saving_template))
        except Exception as e_load_tmpl:
            L.error(f"Slab {slab_idx}: Failed to load AC-PC NIfTI template {example_acpc_slab_nifti_path_for_saving_template}: {e_load_tmpl}. Skipping."); continue

        material_acpc_slab_volumes_to_process: Dict[str, Path] = {} # Holds ACPC material NIfTIs

        if not args.skip_outer_csf:
            vol_outer_csf_acpc = vol_subtract_numpy(m_bm_acpc, m_pc_acpc)
            p_acpc = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-OuterCSF_acpc.nii.gz"
            if save_numpy_as_nifti(vol_outer_csf_acpc, acpc_slab_template_img_for_saving, p_acpc, L): material_acpc_slab_volumes_to_process["OuterCSF"] = p_acpc
        
        vol_gm_acpc = vol_subtract_numpy(m_pc_acpc, m_wc_acpc)
        p_acpc = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-GreyMatter_acpc.nii.gz"
        if save_numpy_as_nifti(vol_gm_acpc, acpc_slab_template_img_for_saving, p_acpc, L): material_acpc_slab_volumes_to_process["GreyMatter"] = p_acpc

        vol_vent_final_acpc = m_vent_acpc.copy() if m_vent_acpc is not None else zeros_for_fallback_acpc_slab.copy()
        p_acpc_vent = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-Ventricles_acpc.nii.gz"
        if save_numpy_as_nifti(vol_vent_final_acpc, acpc_slab_template_img_for_saving, p_acpc_vent, L): material_acpc_slab_volumes_to_process["Ventricles"] = p_acpc_vent

        vol_sgm_final_acpc = m_sgm_acpc.copy() if m_sgm_acpc is not None else zeros_for_fallback_acpc_slab.copy()
        p_acpc_sgm = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-SubcorticalGrey_acpc.nii.gz"
        if save_numpy_as_nifti(vol_sgm_final_acpc, acpc_slab_template_img_for_saving, p_acpc_sgm, L): material_acpc_slab_volumes_to_process["SubcorticalGrey"] = p_acpc_sgm
        
        working_wm_acpc = m_wc_acpc.copy() if m_wc_acpc is not None else zeros_for_fallback_acpc_slab.copy()
        working_wm_acpc = vol_subtract_numpy(working_wm_acpc, vol_vent_final_acpc)
        working_wm_acpc = vol_subtract_numpy(working_wm_acpc, vol_sgm_final_acpc)
        vol_wm_final_acpc = working_wm_acpc
        p_acpc_wm = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-WhiteMatter_acpc.nii.gz"
        if save_numpy_as_nifti(vol_wm_final_acpc, acpc_slab_template_img_for_saving, p_acpc_wm, L): material_acpc_slab_volumes_to_process["WhiteMatter"] = p_acpc_wm
        runlog["steps"].append(f"Slab {slab_idx}: Volumetric material definition in AC-PC space completed.")


        # --- Step 8: Resample AC-PC Material Slabs to Native T1w Space (ALWAYS performed before meshing) ---
        L.info(f"--- Slab {slab_idx}: Resampling AC-PC material slabs to native T1w space for meshing ---")
        final_material_volumes_to_mesh: Dict[str, Path] = {} # This will hold paths to NATIVE T1w NIfTIs for meshing

        native_space_reference_for_final_resampling = actual_flirt_src_path
        L.info(f"Using '{native_space_reference_for_final_resampling.name}' as the reference grid for native space resampling (Step 8).")

        for mat_name, acpc_vol_path in material_acpc_slab_volumes_to_process.items():
            native_vol_path = material_slabs_native_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{mat_name}_native_resampled.nii.gz"
            if resample_volume_to_native_space(
                acpc_vol_path, native_vol_path, native_space_reference_for_final_resampling, 
                acpc_to_native_fsl_mat_path, 
                L, interpolation="trilinear", verbose=args.verbose
            ):
                final_material_volumes_to_mesh[mat_name] = native_vol_path
            else:
                L.warning(f"Failed to resample {mat_name} for slab {slab_idx} to native space. Skipping meshing for this material.")
        runlog["steps"].append(f"Slab {slab_idx}: Resampling of materials to native space completed.")

        if not final_material_volumes_to_mesh:
            L.warning(f"Slab {slab_idx}: No material volumes successfully resampled to native space. Skipping meshing and flattening for this slab.")
            continue

        # --- Step 9: Meshing Final Material Slab Volumes (from Native Space) & Optional Cap Flattening ---
        final_output_space_str = "ACPC" if args.output_final_slabs_in_acpc_space else "Native"
        L.info(f"--- Slab {slab_idx}: Meshing native-space material slab volumes. Final output space for STLs: {final_output_space_str} ---")
        
        material_vol_names_for_mesh = ["OuterCSF", "GreyMatter", "WhiteMatter", "Ventricles", "SubcorticalGrey"]
        if args.skip_outer_csf and "OuterCSF" in material_vol_names_for_mesh: material_vol_names_for_mesh.remove("OuterCSF")

        ideal_cap_coords_world_for_slab: Optional[Tuple[float, float]] = None
        current_slab_def_info = next((item for item in slab_definitions_voxel_acpc if item["slab_idx"] == slab_idx), None)

        # Determine if cap flattening can proceed for this slab
        can_flatten_this_slab = (
            args.flatten_caps_tolerance_mm is not None and
            native_to_acpc_world_affine is not None and # Needed to transform native mesh to ACPC
            (acpc_to_native_world_affine is not None or args.output_final_slabs_in_acpc_space) and # Needed if output is native
            cropped_acpc_grid_affine_for_caps is not None and
            current_slab_def_info is not None
        )

        if can_flatten_this_slab:
            ideal_cap_coords_world_for_slab = get_ideal_cap_coords_world_acpc(
                slab_def_vox_start=current_slab_def_info["start_vox"],
                slab_def_vox_thickness=current_slab_def_info["thickness_vox"],
                slice_axis_idx_acpc=slice_axis_idx,
                cropped_acpc_grid_affine=cropped_acpc_grid_affine_for_caps, 
                logger=L
            )
            if ideal_cap_coords_world_for_slab:
                 L.info(f"Slab {slab_idx}: Target Ideal AC-PC World Cap Coords (axis {slice_axis_idx}): "
                        f"Min={ideal_cap_coords_world_for_slab[0]:.4f}, Max={ideal_cap_coords_world_for_slab[1]:.4f}")
            else:
                L.warning(f"Slab {slab_idx}: Could not calculate ideal cap coordinates. Disabling flattening for this slab's materials.")
                can_flatten_this_slab = False 
        elif args.flatten_caps_tolerance_mm is not None: 
            L.warning(f"Slab {slab_idx}: Prerequisites for cap flattening not met (e.g. missing affines). Skipping for this slab's materials.")


        for mat_name in material_vol_names_for_mesh:
            native_vol_path_for_mesh = final_material_volumes_to_mesh.get(mat_name)
            if native_vol_path_for_mesh and native_vol_path_for_mesh.exists():
                # Filename reflects the *final* output space.
                stl_path = stl_output_base_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{mat_name}_material_space-{final_output_space_str.lower()}.stl"
                
                # vol_to_mesh is called on NATIVE volumes, so mesh_generated is NATIVE
                mesh_generated_successfully = vol_to_mesh(native_vol_path_for_mesh, stl_path, args.no_final_mesh_smoothing, L)

                if mesh_generated_successfully and can_flatten_this_slab and ideal_cap_coords_world_for_slab:
                    L.info(f"Attempting to flatten caps for {stl_path.name} (Slab {slab_idx}, Mat {mat_name}). Final output space: {final_output_space_str}")
                    
                    try:
                        # Load the NATIVE mesh that was just saved by vol_to_mesh
                        native_mesh_for_flattening = trimesh.load_mesh(str(stl_path)) 
                        if not native_mesh_for_flattening.is_empty:
                            ideal_coord_min, ideal_coord_max = ideal_cap_coords_world_for_slab
                            
                            flattened_mesh_final_space = flatten_mesh_caps_acpc(
                                mesh=native_mesh_for_flattening, # Always pass native mesh here
                                native_to_acpc_world_affine=native_to_acpc_world_affine, 
                                acpc_to_native_world_affine=acpc_to_native_world_affine, 
                                ideal_coord_min_acpc=ideal_coord_min,
                                ideal_coord_max_acpc=ideal_coord_max,
                                slice_axis_idx_acpc=slice_axis_idx,
                                tolerance_mm=args.flatten_caps_tolerance_mm, 
                                logger=L,
                                output_space_is_acpc=args.output_final_slabs_in_acpc_space # This controls output of flatten_mesh_caps_acpc
                            )
                            if flattened_mesh_final_space and not flattened_mesh_final_space.is_empty:
                                if not np.array_equal(flattened_mesh_final_space.vertices, native_mesh_for_flattening.vertices) or args.output_final_slabs_in_acpc_space : # Check if vertices changed OR if space changed
                                    L.info(f"Caps/Space modified for {stl_path.name}. Exporting updated version.")
                                    flattened_mesh_final_space.export(str(stl_path)) # Overwrite with final space mesh
                                    runlog["steps"].append(f"Caps flattened and mesh saved in {final_output_space_str} for {mat_name} slab {slab_idx}")
                                else:
                                    L.info(f"Flattening did not change vertices for {stl_path.name} and output is native. Original native mesh retained.")
                            elif flattened_mesh_final_space is None or flattened_mesh_final_space.is_empty: 
                                L.warning(f"Cap flattening returned None or an empty mesh for {stl_path.name}. Original (native) mesh retained.")
                        else:
                            L.warning(f"Loaded mesh for flattening {stl_path.name} is empty. Skipping flattening.")
                    except FileNotFoundError:
                        L.error(f"Mesh file {stl_path.name} not found for loading to flatten caps.")
                    except Exception as e_flatten:
                        L.error(f"Error during cap flattening for {stl_path.name}: {e_flatten}. Original (native) mesh retained.", exc_info=True)
                        runlog["warnings"].append(f"Cap flattening failed for {mat_name} slab {slab_idx}: {str(e_flatten)[:100]}")

                if stl_path.exists() and stl_path.stat().st_size > 0:
                    runlog["output_files"].append(str(stl_path))
                elif mesh_generated_successfully: 
                     L.warning(f"Mesh {stl_path.name} was generated but is now missing or empty post-flattening attempt.")
                     runlog["warnings"].append(f"Mesh {stl_path.name} missing/empty post-flattening attempt.")
            else:
                L.warning(f"Native space volume for {mat_name} slab {slab_idx} not found or not saved. Skipping mesh generation.")
        runlog["steps"].append(f"Slab {slab_idx}: Meshing of final materials from native space completed. Output STLs in {final_output_space_str} space.")
    L.info("Multi-material slab component generation workflow finished successfully.")
    return True

def main():
    args = _build_parser().parse_args()
    script_name_stem = Path(__file__).stem
    L_main = get_logger(script_name_stem, level=logging.DEBUG if args.verbose else logging.INFO)
    _L_HELPERS.setLevel(L_main.getEffectiveLevel()) 

    runlog: Dict[str, Any] = {
        "tool": script_name_stem,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if v is not None},
        "steps": [], "warnings": [],
        "output_dir": os.path.abspath(args.output_dir),
        "output_files": []
    }
    
    if args.mni_template:
        args.mni_template = Path(args.mni_template).resolve()
        if not args.mni_template.exists():
            L_main.error(f"Specified MNI template path does not exist: {args.mni_template}")
            sys.exit(1)
    elif DEFAULT_MNI_TEMPLATE_PATH and DEFAULT_MNI_TEMPLATE_PATH.exists():
        args.mni_template = DEFAULT_MNI_TEMPLATE_PATH.resolve()
        L_main.info(f"Using MNI template for AC-PC alignment from FSLDIR: {args.mni_template}")
    else:
        L_main.error(f"Default MNI template path could not be determined or does not exist ({DEFAULT_MNI_TEMPLATE_PATH}). "
                     "Ensure FSLDIR is set and FSL is installed correctly, or provide --mni_template.")
        sys.exit(1)


    final_stl_output_dir = Path(args.output_dir).resolve()
    final_stl_output_dir.mkdir(parents=True, exist_ok=True)
    success = False

    if args.work_dir:
        work_dir_path = Path(args.work_dir).resolve()
        work_dir_path.mkdir(parents=True, exist_ok=True)
        L_main.info(f"Using specified work directory: {work_dir_path}")
        runlog["steps"].append(f"Using specified work directory: {work_dir_path}")
        try:
            success = main_wf(args, L_main, work_dir_path, final_stl_output_dir, runlog)
            if args.no_clean : L_main.info("--no_clean is active. Retaining work directory.") 
        except Exception as e:
            L_main.error(f"An error occurred in main_wf with specified work_dir: {e}", exc_info=True)
            runlog["warnings"].append(f"CRITICAL_ERROR: {str(e)}")
            success = False
    else:
        temp_base_dir = Path(args.output_dir).resolve() 
        temp_dir_context = temp_dir(tag=f"{script_name_stem}_work", keep=args.no_clean, base_dir=str(temp_base_dir))
        try:
            with temp_dir_context as temp_d_str:
                temp_work_dir_path = Path(temp_d_str)
                L_main.info(f"Using temporary work directory: {temp_work_dir_path}")
                runlog["steps"].append(f"Using temporary work directory: {temp_work_dir_path}")
                success = main_wf(args, L_main, temp_work_dir_path, final_stl_output_dir, runlog)
                if args.no_clean and temp_work_dir_path.exists(): 
                     runlog["warnings"].append(f"Temporary work directory retained: {temp_work_dir_path}")
                     L_main.warning(f"Temporary work directory retained by --no_clean: {temp_work_dir_path}")
        except Exception as e:
            L_main.error(f"An error occurred during temporary directory handling or main workflow: {e}", exc_info=True)
            runlog["warnings"].append(f"CRITICAL_ERROR: {str(e)}")
            success = False
    
    if "args" in runlog:
        for key, value in runlog["args"].items():
            if isinstance(value, Path): runlog["args"][key] = str(value)
    
    write_log(runlog, str(final_stl_output_dir), base_name=f"{script_name_stem}_log")

    if success: L_main.info(f"{script_name_stem} finished successfully.")
    else: L_main.error(f"{script_name_stem} failed. See logs for details."); sys.exit(1)

if __name__ == "__main__":
    main()

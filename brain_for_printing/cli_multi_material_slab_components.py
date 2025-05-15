#!/usr/bin/env python
# brain_for_printing/cli_multi_material_slab_components.py

"""
Generates distinct, non-overlapping multi-material brain slab components
suitable for 3D printing, with an initial AC-PC alignment, volumetric slicing,
and a reverse transform to native space before meshing to help cap slabs.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import shutil 

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

# --- Helper Functions ---

def acpc_align_t1w(
    input_t1w_path: Path,
    output_acpc_t1w_path: Path,
    output_transform_mat_path: Path, 
    mni_template_path: Path,
    temp_dir_path: Path, 
    logger: logging.Logger,
    verbose: bool = False
) -> bool:
    logger.info(f"Starting AC-PC alignment for {input_t1w_path.name} -> {output_acpc_t1w_path.name}")
    require_cmds(["flirt", "robustfov"], logger=logger)
    if not mni_template_path.exists():
        logger.error(f"MNI template not found at {mni_template_path}.")
        return False

    t1w_cropped_path = temp_dir_path / f"{input_t1w_path.stem}_croppedfov.nii.gz"
    robustfov_cmd = ["robustfov", "-i", str(input_t1w_path), "-r", str(t1w_cropped_path)]
    t1w_to_flirt = input_t1w_path 
    try:
        logger.debug(f"Running robustfov command: {' '.join(robustfov_cmd)}")
        run_cmd(robustfov_cmd, verbose=verbose)
        if t1w_cropped_path.exists() and t1w_cropped_path.stat().st_size > 0:
            t1w_to_flirt = t1w_cropped_path
            logger.info(f"robustfov completed, using {t1w_cropped_path.name} for FLIRT.")
        else:
            logger.warning(f"robustfov output not created or empty. Using original T1w for FLIRT: {input_t1w_path.name}")
    except Exception as e_fov:
        logger.warning(f"robustfov failed: {e_fov}. Using original T1w for FLIRT: {input_t1w_path.name}", exc_info=verbose)

    flirt_cmd = [
        "flirt", "-in", str(t1w_to_flirt), "-ref", str(mni_template_path),
        "-out", str(output_acpc_t1w_path), "-omat", str(output_transform_mat_path), 
        "-dof", "6", "-cost", "corratio",
        "-searchrx", "-90", "90", "-searchry", "-90", "90", "-searchrz", "-90", "90",
        "-interp", "trilinear"
    ]
    try:
        logger.debug(f"Running FLIRT command: {' '.join(flirt_cmd)}")
        run_cmd(flirt_cmd, verbose=verbose)
        if output_acpc_t1w_path.exists() and output_acpc_t1w_path.stat().st_size > 0 and \
           output_transform_mat_path.exists() and output_transform_mat_path.stat().st_size > 0 :
            logger.info(f"Successfully created AC-PC aligned T1w: {output_acpc_t1w_path.name} and transform: {output_transform_mat_path.name}")
            if t1w_to_flirt != input_t1w_path and t1w_to_flirt.exists():
                t1w_to_flirt.unlink(missing_ok=True)
            return True
        else:
            logger.error(f"FLIRT ran but output AC-PC aligned T1w or transform not created/empty.")
            return False
    except Exception as e:
        logger.error(f"AC-PC alignment using FLIRT failed: {e}", exc_info=verbose)
        return False

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
    cmd = [
        "flirt",
        "-in", str(input_acpc_slab_volume_path),
        "-ref", str(original_native_t1w_ref_path),
        "-applyxfm",
        "-init", str(acpc_to_native_t1w_mat_path),
        "-out", str(output_native_slab_volume_path),
        "-interp", interpolation # Crucial for creating partial volumes at edges
    ]
    try:
        logger.debug(f"Running FLIRT for resampling to native: {' '.join(cmd)}")
        run_cmd(cmd, verbose=verbose)
        if output_native_slab_volume_path.exists() and output_native_slab_volume_path.stat().st_size > 0:
            logger.info(f"Successfully resampled to native space: {output_native_slab_volume_path.name}")
            return True
        else:
            logger.error(f"FLIRT resampling ran but output native volume not created or empty.")
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
                     f"Calculated: x={x_min},{x_size}; y={y_min},{y_size}; z={z_min},{z_size}"); return False
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
        # For resampled data (which might have float values), check max value.
        # For binary masks, sum check is okay.
        is_effectively_empty = False
        if data_check.dtype == np.uint8 or data_check.dtype == bool: # Binary mask
            if data_check.size > 0 and np.sum(data_check) == 0:
                is_effectively_empty = True
        elif data_check.size > 0 and np.max(data_check) < 0.1: # Heuristic for float PV maps
             is_effectively_empty = True
        
        if is_effectively_empty:
            logger.info(f"Volume {volume_path.name} appears effectively empty. Skipping mesh generation.")
            return False
            
    except Exception as e:
        logger.error(f"Could not load volume {volume_path.name} to check if empty: {e}", exc_info=True)
        return False

    temp_gii_path = output_mesh_path.with_name(f"{output_mesh_path.stem}_{os.urandom(4).hex()}.surf.gii")
    mesh: Optional[trimesh.Trimesh] = None
    try:
        # Marching cubes level should be 0.5 for binary masks,
        # and also for partial volume maps where 0.5 represents the isosurface.
        volume_to_gifti(str(volume_path), str(temp_gii_path), level=0.5) 
        if temp_gii_path.exists() and temp_gii_path.stat().st_size > 0:
            mesh = gifti_to_trimesh(str(temp_gii_path))
            logger.debug(f"After gifti_to_trimesh for {volume_path.name}, type(mesh) is {type(mesh)}")
        else:
            logger.warning(f"GIFTI file {temp_gii_path.name} was not created or is empty from {volume_path.name}.")
    except Exception as e_vtg:
        logger.error(f"volume_to_gifti or gifti_to_trimesh failed for {volume_path.name}: {e_vtg}", exc_info=True)
    finally:
        if temp_gii_path.exists(): temp_gii_path.unlink(missing_ok=True)

    if not isinstance(mesh, trimesh.Trimesh) or (hasattr(mesh, 'is_empty') and mesh.is_empty):
        logger.warning(f"Mesh from {volume_path.name} is not a valid Trimesh object or is empty after gifti step. Type: {type(mesh)}")
        return False
    logger.debug(f"Initial mesh for {output_mesh_path.name}: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces. Watertight: {mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'}")

    try:
        logger.debug(f"Processing mesh for {output_mesh_path.name} with validate=True...")
        mesh.process(validate=True)
        logger.debug(f"After mesh.process for {output_mesh_path.name}, Watertight: {mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'}")
        if not isinstance(mesh, trimesh.Trimesh):
             logger.error(f"CRITICAL: mesh became {type(mesh)} after mesh.process() for {output_mesh_path.name}!")
             return False
    except Exception as e_process:
        logger.warning(f"mesh.process() failed for {output_mesh_path.name}: {e_process}. Proceeding.", exc_info=True)

    try:
        if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
            logger.debug(f"Attempting to fill holes for {output_mesh_path.name}...")
            mesh.fill_holes() 
            logger.info(f"After fill_holes for {output_mesh_path.name}, Watertight: {mesh.is_watertight if hasattr(mesh, 'is_watertight') else 'N/A'}") # Changed to INFO
            if not isinstance(mesh, trimesh.Trimesh):
                 logger.error(f"CRITICAL: mesh became {type(mesh)} after fill_holes for {output_mesh_path.name}!")
                 return False
    except Exception as e_fill:
        logger.warning(f"Attempting to fill_holes on {output_mesh_path.name} failed: {e_fill}. Proceeding.", exc_info=True)

    if not no_smooth:
        logger.info(f"Attempting Taubin smoothing for {output_mesh_path.name}...")
        try:
            if isinstance(mesh, trimesh.Trimesh) and \
               hasattr(mesh, 'vertices') and hasattr(mesh, 'faces') and \
               mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0:
                
                original_mesh_pre_smooth_copy = mesh.copy()
                smoothed_mesh_candidate = trimesh.smoothing.filter_taubin(original_mesh_pre_smooth_copy, iterations=10)
                logger.debug(f"After filter_taubin for {output_mesh_path.name}, type(smoothed_mesh_candidate) is {type(smoothed_mesh_candidate)}")

                if isinstance(smoothed_mesh_candidate, trimesh.Trimesh) and not smoothed_mesh_candidate.is_empty:
                    mesh = smoothed_mesh_candidate
                    logger.debug(f"Successfully applied Taubin smoothing to {output_mesh_path.name}.")
                else:
                    logger.warning(
                        f"Taubin smoothing for {output_mesh_path.name} did not return a valid, non-empty Trimesh object "
                        f"(got type: {type(smoothed_mesh_candidate)}). Using mesh state from before smoothing attempt."
                    )
            else:
                logger.warning(f"Skipping Taubin smoothing for {output_mesh_path.name} as it's not a valid Trimesh with vertices/faces before smoothing. Type: {type(mesh)}")
        except Exception as e_smooth:
            logger.warning(f"Taubin smoothing process failed for {output_mesh_path.name}: {e_smooth}. Using mesh state from before smoothing attempt.", exc_info=True)
        
        logger.debug(f"After smoothing block for {output_mesh_path.name}, type(mesh) is {type(mesh)}")
        if not isinstance(mesh, trimesh.Trimesh):
            logger.error(f"CRITICAL: mesh became {type(mesh)} after smoothing block for {output_mesh_path.name}!")
            return False

    try:
        if isinstance(mesh, trimesh.Trimesh):
            logger.debug(f"Attempting to fix normals for {output_mesh_path.name}...")
            mesh.fix_normals(multibody=True) 
            logger.debug(f"After fix_normals for {output_mesh_path.name}, type(mesh) is {type(mesh)}")
            if not isinstance(mesh, trimesh.Trimesh):
                logger.error(f"CRITICAL: mesh became {type(mesh)} after fix_normals for {output_mesh_path.name}!")
                return False
        else:
            logger.warning(f"Skipping fix_normals for {output_mesh_path.name} because mesh is not a Trimesh object (type: {type(mesh)}).")
    except Exception as e_norm:
        logger.warning(f"Fixing normals failed for {output_mesh_path.name}: {e_norm}. Mesh might have issues.", exc_info=True)

    logger.debug(f"Preparing to export {output_mesh_path.name}. Mesh final check - Type: {type(mesh)}, Is Trimesh: {isinstance(mesh, trimesh.Trimesh)}, Watertight: {mesh.is_watertight if hasattr(mesh,'is_watertight') else 'N/A'}")
    if not isinstance(mesh, trimesh.Trimesh):
        logger.error(f"CRITICAL: 'mesh' variable is NOT a Trimesh object before export for {output_mesh_path.name}. Type: {type(mesh)}")
        return False

    try:
        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_mesh_path)) 
        if output_mesh_path.exists() and output_mesh_path.stat().st_size > 0:
             logger.info(f"Successfully exported mesh: {output_mesh_path}")
             return True
        else: 
             logger.error(f"Mesh export command seemed to run for {output_mesh_path.name} but file is missing or empty.")
             return False
    except AttributeError as e_attrib_export:
        logger.error(f"AttributeError during export of {output_mesh_path.name}: {e_attrib_export}. 'mesh' type: {type(mesh)}", exc_info=True)
        return False
    except Exception as e_export:
        logger.error(f"Failed to export mesh {output_mesh_path.name}: {e_export}", exc_info=True)
        return False

# --- Argument Parser ---
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
    return parser

# --- Main Workflow ---
def main_wf(args: argparse.Namespace, L: logging.Logger, work_dir: Path, stl_output_base_dir: Path, runlog: Dict[str, Any]):
    runlog["steps"].append("Starting multi-material slab component generation with AC-PC alignment and volumetric slicing.")
    L.info(f"Target AC-PC aligned voxel resolution: {args.voxel_resolution} mm")
    L.info(f"Slab thickness: {args.slab_thickness}mm, Orientation (AC-PC): {args.slab_orientation}")
    L.info(f"BrainMask inflation: {args.brain_mask_inflate_mm}mm")

    mrtrix_cmds = ["mrgrid", "mesh2voxel"]
    fsl_cmds = ["flirt", "fslroi", "robustfov", "convert_xfm"]
    require_cmds(mrtrix_cmds + fsl_cmds, logger=L)
    runlog["steps"].append(f"Required commands verified: {', '.join(mrtrix_cmds + fsl_cmds)}")
    
    acpc_align_dir = work_dir / "00_acpc_alignment"
    parent_surf_gen_dir = work_dir / "01_parent_surface_generation"
    full_vol_voxelized_dir = work_dir / "02_full_acpc_voxelized_pvs"
    full_vol_binarized_dir = work_dir / "03_full_acpc_binarized_masks"
    cropped_full_vol_dir = work_dir / "04_cropped_full_acpc_masks" 
    volumetric_slabs_acpc_dir = work_dir / "05_volumetric_slabs_acpc_nifti"
    material_slabs_acpc_vol_dir = work_dir / "06_material_slabs_acpc_volumes_nifti"
    material_slabs_native_vol_dir = work_dir / "07_material_slabs_native_volumes_nifti" # NEW directory
    
    for d_path in [acpc_align_dir, parent_surf_gen_dir, full_vol_voxelized_dir, 
                   full_vol_binarized_dir, cropped_full_vol_dir, volumetric_slabs_acpc_dir, 
                   material_slabs_acpc_vol_dir, material_slabs_native_vol_dir]: # Added new dir
        d_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: AC-PC Alignment of T1w & Inverse Transform ---
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
    t1w_to_acpc_mat_path = acpc_align_dir / f"{original_t1w_path_for_ref.stem}_to_acpc.mat"
    mni_template_for_acpc = Path(args.mni_template) 
        
    if not acpc_align_t1w(original_t1w_path_for_ref, t1w_acpc_aligned_path, t1w_to_acpc_mat_path, 
                          mni_template_for_acpc, acpc_align_dir, L, args.verbose):
        L.error("AC-PC alignment step failed."); return False
    runlog["steps"].append(f"AC-PC T1w: {t1w_acpc_aligned_path.name}, XFM: {t1w_to_acpc_mat_path.name}")

    acpc_to_native_t1w_mat_path = acpc_align_dir / f"{original_t1w_path_for_ref.stem}_acpc_to_native.mat"
    if not invert_fsl_transform(t1w_to_acpc_mat_path, acpc_to_native_t1w_mat_path, L, args.verbose):
        L.error("Failed to invert AC-PC alignment transform."); return False
    runlog["steps"].append(f"Inverse XFM (ACPC->Native): {acpc_to_native_t1w_mat_path.name}")

    # --- Step 2: Create High-Resolution Master AC-PC Template ---
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

    # --- Step 3: Generate Full Parent Surfaces ---
    L.info("--- Step 3: Generating full parent T1-space surfaces (Trimesh objects) ---")
    # (This section is identical to previous version, using args.brain_mask_inflate_mm)
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
    L.debug(f"PialComplex - Other structures to generate: {pial_other_explicit_for_pial_complex}")
    pial_comp_meshes = generate_brain_surfaces(
        args.subjects_dir, args.subject_id, "T1", 
        tuple(pial_cortical_for_complex), list(pial_other_explicit_for_pial_complex),       
        [], [], args.run, args.session, args.verbose, str(parent_surf_gen_dir / "pial_complex_work"))
    for k, v_mesh in pial_comp_meshes.items():
        if v_mesh and not v_mesh.is_empty: parent_meshes_trimesh[k] = v_mesh 
        else: L.warning(f"Pial complex component '{k}' generation failed or resulted in an empty mesh.")

    L.info("Generating White Complex components...")
    white_cortical_for_complex, white_other_for_complex_from_preset, _ = parse_preset("white_brain")
    white_other_explicit_for_white_complex = set()
    for item in white_other_for_complex_from_preset:
        if item == "cerebellum_wm": white_other_explicit_for_white_complex.add("cerebellum_wm") 
        elif item == "cerebellum": 
            L.warning("'cerebellum' found in white_brain preset, interpreting as 'cerebellum_wm' for WhiteComplex.")
            white_other_explicit_for_white_complex.add("cerebellum_wm")
        else: white_other_explicit_for_white_complex.add(item)
    L.debug(f"WhiteComplex - Other structures to generate: {white_other_explicit_for_white_complex}")
    white_comp_meshes = generate_brain_surfaces(
        args.subjects_dir, args.subject_id, "T1", 
        tuple(white_cortical_for_complex), list(white_other_explicit_for_white_complex),      
        [], [], args.run, args.session, args.verbose, str(parent_surf_gen_dir / "white_complex_work"))
    for k, v_mesh in white_comp_meshes.items():
        if v_mesh and not v_mesh.is_empty: parent_meshes_trimesh[k] = v_mesh
        else: L.warning(f"White complex component '{k}' generation failed or resulted in an empty mesh.")

    if is_vtk_available():
        fs_input_dir = Path(args.subjects_dir) / "sourcedata" / "freesurfer" / args.subject_id
        if fs_input_dir.is_dir():
            five_tt_gen_dir = parent_surf_gen_dir / "5ttgen_work_multimat" 
            five_tt_gen_dir.mkdir(parents=True, exist_ok=True)
            if run_5ttgen_hsvs_save_temp_bids(args.subject_id, str(fs_input_dir), str(five_tt_gen_dir), args.session, verbose=args.verbose):
                vtk_meshes = load_subcortical_and_ventricle_meshes(str(five_tt_gen_dir))
                for k, v_mesh in vtk_meshes.items(): 
                    if v_mesh and not v_mesh.is_empty: parent_meshes_trimesh[k] = v_mesh
                    else: L.debug(f"VTK-derived mesh '{k}' was empty or failed to load, not adding.")
            else: L.warning("5ttgen command failed, SGM/Ventricles might be missing.")
        else: L.warning(f"FreeSurfer dir for 5ttgen not found: {fs_input_dir}, skipping SGM/Ventricles.")
    else: L.warning("VTK not available, skipping SGM/Ventricles.")
    
    valid_parent_meshes_trimesh = {k:v for k,v in parent_meshes_trimesh.items() if v and not v.is_empty}
    if not valid_parent_meshes_trimesh: L.error("No valid parent Trimesh surfaces generated. Aborting."); return False
    L.info(f"Generated {len(valid_parent_meshes_trimesh)} initial parent Trimesh surfaces: {list(valid_parent_meshes_trimesh.keys())}")

    # --- Step 4: Voxelize Full Parent Surfaces onto Master AC-PC Template & Binarize ---
    L.info(f"--- Step 4: Voxelizing full parent surfaces onto Master AC-PC Template ({master_hires_acpc_template_path.name}) ---")
    full_acpc_binarized_masks: Dict[str, Path] = {}
    temp_obj_export_dir = parent_surf_gen_dir / "temp_full_surf_objs" 
    temp_obj_export_dir.mkdir(parents=True, exist_ok=True)
    for parent_name, trimesh_obj in valid_parent_meshes_trimesh.items():
        safe_name = parent_name.replace("_", "-").replace(" ", "-")
        temp_full_surf_obj_path = temp_obj_export_dir / f"{args.subject_id}_desc-{safe_name}_full_surf.obj"
        try: trimesh_obj.export(str(temp_full_surf_obj_path))
        except Exception as e: L.error(f"Failed to export full {parent_name} to OBJ: {e}", exc_info=True); continue
        full_pv_path = full_vol_voxelized_dir / f"{args.subject_id}_desc-{safe_name}_full_acpc_pv.nii.gz"
        if not mesh_to_partial_volume(temp_full_surf_obj_path, master_hires_acpc_template_path, full_pv_path, L, args.verbose):
            L.warning(f"Failed to voxelize full {parent_name}."); continue
        full_bin_path = full_vol_binarized_dir / f"{args.subject_id}_desc-{safe_name}_full_acpc_mask.nii.gz"
        if binarize_volume_file(full_pv_path, full_bin_path, args.pv_threshold, L):
            full_acpc_binarized_masks[parent_name] = full_bin_path
        else: L.warning(f"Failed to binarize full PV for {parent_name}.")
        if not args.no_clean: temp_full_surf_obj_path.unlink(missing_ok=True)
    if not args.no_clean and temp_obj_export_dir.exists(): shutil.rmtree(temp_obj_export_dir)
    if not full_acpc_binarized_masks: L.error("No full surfaces voxelized/binarized. Aborting."); return False
    L.info(f"Voxelized and binarized {len(full_acpc_binarized_masks)} full parent surfaces onto AC-PC grid.")

    # --- Step 5: Volumetric Cropping ---
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
    for struct_name, full_mask_path in full_acpc_binarized_masks.items():
        safe_name = struct_name.replace("_", "-").replace(" ", "-")
        cropped_path = cropped_full_vol_dir / f"{args.subject_id}_desc-{safe_name}_full_acpc_mask_cropped.nii.gz"
        if crop_nifti_volume_fslroi(full_mask_path, cropped_path, overall_min_coords_vox, overall_max_coords_vox, L, args.verbose):
            cropped_acpc_binarized_masks[struct_name] = cropped_path
        else: L.warning(f"Failed to crop {full_mask_path.name}.")
    if not cropped_acpc_binarized_masks: L.error("No volumes successfully cropped. Aborting."); return False
    L.info(f"Cropped {len(cropped_acpc_binarized_masks)} full AC-PC volumes.")

    # --- Step 6: Volumetric Slicing (AC-PC space) ---
    L.info("--- Step 6: Performing volumetric slicing (in AC-PC space) ---")
    orientation_map = {"axial": 2, "coronal": 1, "sagittal": 0} 
    slice_axis_idx = orientation_map[args.slab_orientation]
    example_cropped_vol_path = next(iter(cropped_acpc_binarized_masks.values()), None)
    if not example_cropped_vol_path: L.error("No cropped volumes for slicing. Aborting."); return False
    try:
        example_cropped_img = nib.load(str(example_cropped_vol_path))
        cropped_vol_dims_xyz = np.array(example_cropped_img.shape[:3], dtype=int)
    except Exception as e: L.error(f"Could not load {example_cropped_vol_path} for slicing dims: {e}", exc_info=True); return False

    slicing_axis_voxel_size_mm = master_template_hires_voxel_sizes[slice_axis_idx]
    slab_thickness_voxels = max(1, int(round(args.slab_thickness / slicing_axis_voxel_size_mm)))
    L.info(f"Slicing axis {slice_axis_idx}. Voxel size: {slicing_axis_voxel_size_mm:.3f}mm. Slab: {args.slab_thickness}mm => {slab_thickness_voxels} voxels.")
    
    num_slabs_generated = 0
    current_slab_start_voxel_in_cropped = 0
    volumetric_slabs_acpc_struct_files: Dict[int, Dict[str, Path]] = {} 
    total_extent_cropped_vox = cropped_vol_dims_xyz[slice_axis_idx]
    while current_slab_start_voxel_in_cropped < total_extent_cropped_vox:
        slab_idx = num_slabs_generated
        volumetric_slabs_acpc_struct_files[slab_idx] = {}
        actual_slab_thickness_this_iteration_vox = min(slab_thickness_voxels, total_extent_cropped_vox - current_slab_start_voxel_in_cropped)
        if actual_slab_thickness_this_iteration_vox <= 0: break
        L.info(f"--- >> Volumetrically Slicing AC-PC Slab Index {slab_idx} (voxels {current_slab_start_voxel_in_cropped} to {current_slab_start_voxel_in_cropped + actual_slab_thickness_this_iteration_vox -1} on axis {slice_axis_idx} of cropped volume) << ---")
        for struct_name, cropped_vol_path in cropped_acpc_binarized_masks.items():
            safe_name = struct_name.replace("_", "-").replace(" ", "-")
            slab_nifti_path = volumetric_slabs_acpc_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{safe_name}_acpc_volslab.nii.gz"
            if extract_volumetric_slab_fslroi(cropped_vol_path, slab_nifti_path, f"slab-{slab_idx}", slice_axis_idx,
                current_slab_start_voxel_in_cropped, actual_slab_thickness_this_iteration_vox,
                tuple(cropped_vol_dims_xyz), L, args.verbose):
                volumetric_slabs_acpc_struct_files[slab_idx][struct_name] = slab_nifti_path
            else: L.warning(f"Failed to extract AC-PC volumetric slab for {struct_name}, slab {slab_idx}.")
        current_slab_start_voxel_in_cropped += actual_slab_thickness_this_iteration_vox
        num_slabs_generated += 1
    if num_slabs_generated == 0: L.error("No volumetric AC-PC slabs extracted. Aborting."); return False
    L.info(f"Processed {num_slabs_generated} volumetric AC-PC slabs.")

    # --- Main Loop: Per-Slab Processing ---
    for slab_idx in range(num_slabs_generated):
        L.info(f"--- >>> Processing Slab {slab_idx + 1}/{num_slabs_generated} <<< ---")
        
        current_slab_acpc_component_paths = volumetric_slabs_acpc_struct_files.get(slab_idx, {})
        if not current_slab_acpc_component_paths:
            L.warning(f"Slab {slab_idx}: No AC-PC component NIfTI files found. Skipping."); continue

        M_KeyStruct_Slab_i_data_dict: Dict[str, Optional[np.ndarray]] = {}
        example_acpc_slab_nifti_path_for_saving_template = None

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
            L.error(f"Slab {slab_idx}: Essential AC-PC BrainMask missing. Skipping."); continue
        
        zeros_fallback_shape_acpc = M_KeyStruct_Slab_i_data_dict["BrainMask"].shape
        zeros_for_fallback_acpc_slab = np.zeros(zeros_fallback_shape_acpc, dtype=M_KeyStruct_Slab_i_data_dict["BrainMask"].dtype)

        # Corrected logic for vol_union_numpy with fallback
        pial_keys = ["pial_L", "pial_R", "corpus_callosum", "cerebellum", "brainstem"]
        pial_arrs = [load_acpc_slab_nifti(k) for k in pial_keys]
        pial_union_result = vol_union_numpy([a for a in pial_arrs if a is not None])
        M_KeyStruct_Slab_i_data_dict["PialComplex"] = pial_union_result if pial_union_result is not None else zeros_for_fallback_acpc_slab.copy()
        
        white_keys = ["white_L", "white_R", "corpus_callosum", "cerebellum_wm", "brainstem"]
        white_arrs = [load_acpc_slab_nifti(k) for k in white_keys]
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
            L.error(f"Slab {slab_idx}: No example AC-PC slab NIfTI for header info. Skipping save."); continue
        try:
            acpc_slab_template_img_for_saving = nib.load(str(example_acpc_slab_nifti_path_for_saving_template))
        except Exception: L.error(f"Slab {slab_idx}: Failed to load AC-PC slab NIfTI {example_acpc_slab_nifti_path_for_saving_template}. Skipping save."); continue
        
        material_acpc_slab_volumes_to_resample: Dict[str, Path] = {}

        if not args.skip_outer_csf:
            vol_outer_csf_acpc = vol_subtract_numpy(m_bm_acpc, m_pc_acpc)
            p_acpc = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-OuterCSF_acpc.nii.gz"
            if save_numpy_as_nifti(vol_outer_csf_acpc, acpc_slab_template_img_for_saving, p_acpc, L): material_acpc_slab_volumes_to_resample["OuterCSF"] = p_acpc
        
        vol_gm_acpc = vol_subtract_numpy(m_pc_acpc, m_wc_acpc)
        p_acpc = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-GreyMatter_acpc.nii.gz"
        if save_numpy_as_nifti(vol_gm_acpc, acpc_slab_template_img_for_saving, p_acpc, L): material_acpc_slab_volumes_to_resample["GreyMatter"] = p_acpc
        
        vol_vent_final_acpc = m_vent_acpc.copy() 
        p_acpc_vent = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-Ventricles_acpc.nii.gz"
        if save_numpy_as_nifti(vol_vent_final_acpc, acpc_slab_template_img_for_saving, p_acpc_vent, L): material_acpc_slab_volumes_to_resample["Ventricles"] = p_acpc_vent
        else: vol_vent_final_acpc = zeros_for_fallback_acpc_slab.copy()
            
        vol_sgm_final_acpc = m_sgm_acpc.copy()
        p_acpc_sgm = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-SubcorticalGrey_acpc.nii.gz"
        if save_numpy_as_nifti(vol_sgm_final_acpc, acpc_slab_template_img_for_saving, p_acpc_sgm, L): material_acpc_slab_volumes_to_resample["SubcorticalGrey"] = p_acpc_sgm
        else: vol_sgm_final_acpc = zeros_for_fallback_acpc_slab.copy()

        working_wm_acpc = m_wc_acpc.copy()
        working_wm_acpc = vol_subtract_numpy(working_wm_acpc, vol_vent_final_acpc)
        working_wm_acpc = vol_subtract_numpy(working_wm_acpc, vol_sgm_final_acpc)
        vol_wm_final_acpc = working_wm_acpc
        p_acpc_wm = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-WhiteMatter_acpc.nii.gz"
        if save_numpy_as_nifti(vol_wm_final_acpc, acpc_slab_template_img_for_saving, p_acpc_wm, L): material_acpc_slab_volumes_to_resample["WhiteMatter"] = p_acpc_wm
        runlog["steps"].append(f"Slab {slab_idx}: Volumetric material definition in AC-PC space completed.")

        # --- Step 8: Reverse Transform Material Slabs to Native T1w Space ---
        L.info(f"--- Slab {slab_idx}: Resampling AC-PC material slabs to native T1w space ---")
        final_material_native_volumes_to_mesh: Dict[str, Path] = {}
        for mat_name, acpc_vol_path in material_acpc_slab_volumes_to_resample.items():
            native_vol_path = material_slabs_native_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{mat_name}_native_resampled.nii.gz"
            if resample_volume_to_native_space(
                acpc_vol_path, native_vol_path, original_t1w_path_for_ref, 
                acpc_to_native_t1w_mat_path, L, interpolation="trilinear", verbose=args.verbose
            ):
                final_material_native_volumes_to_mesh[mat_name] = native_vol_path
            else:
                L.warning(f"Failed to resample {mat_name} for slab {slab_idx} to native space. Skipping meshing for this material.")
        runlog["steps"].append(f"Slab {slab_idx}: Resampling of materials to native space completed.")

        # --- Step 9: Meshing Final Material Slab Volumes (from Native Space) ---
        L.info(f"--- Slab {slab_idx}: Meshing final native-space material slab volumes to STL ---")
        material_vol_names_for_mesh = ["OuterCSF", "GreyMatter", "WhiteMatter", "Ventricles", "SubcorticalGrey"]
        if args.skip_outer_csf and "OuterCSF" in material_vol_names_for_mesh: material_vol_names_for_mesh.remove("OuterCSF")

        for mat_name in material_vol_names_for_mesh:
            native_vol_path_for_mesh = final_material_native_volumes_to_mesh.get(mat_name)
            if native_vol_path_for_mesh and native_vol_path_for_mesh.exists(): 
                stl_path = stl_output_base_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{mat_name}_material.stl"
                if vol_to_mesh(native_vol_path_for_mesh, stl_path, args.no_final_mesh_smoothing, L):
                    runlog["output_files"].append(str(stl_path))
            else:
                L.warning(f"Native space volume for {mat_name} slab {slab_idx} not found or not saved. Skipping mesh.")
        runlog["steps"].append(f"Slab {slab_idx}: Meshing of final materials from native space completed.")

    L.info("Multi-material slab component generation workflow finished successfully.")
    return True

# --- Main function ---
def main():
    args = _build_parser().parse_args()
    script_name_stem = Path(__file__).stem
    L_main = get_logger(script_name_stem, level=logging.DEBUG if args.verbose else logging.INFO)

    runlog: Dict[str, Any] = { 
        "tool": script_name_stem,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if v is not None},
        "steps": [], "warnings": [],
        "output_dir": os.path.abspath(args.output_dir),
        "output_files": []
    }

    if args.mni_template:
        args.mni_template = Path(args.mni_template)
        if not args.mni_template.exists():
            L_main.error(f"Specified MNI template path does not exist: {args.mni_template}")
            sys.exit(1)
    elif DEFAULT_MNI_TEMPLATE_PATH and DEFAULT_MNI_TEMPLATE_PATH.exists():
        args.mni_template = DEFAULT_MNI_TEMPLATE_PATH
    else: 
        L_main.error(f"Default MNI template path could not be determined or does not exist ({DEFAULT_MNI_TEMPLATE_PATH}). "
                     "Ensure FSLDIR is set and FSL is installed correctly, or provide --mni_template.")
        sys.exit(1)
    L_main.info(f"Using MNI template for AC-PC alignment: {args.mni_template}")

    final_stl_output_dir = Path(args.output_dir) 
    final_stl_output_dir.mkdir(parents=True, exist_ok=True)
    success = False 

    if args.work_dir:
        work_dir_path = Path(args.work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
        L_main.info(f"Using specified work directory: {work_dir_path}")
        runlog["steps"].append(f"Using specified work directory: {work_dir_path}")
        try:
            success = main_wf(args, L_main, work_dir_path, final_stl_output_dir, runlog)
            if args.no_clean : L_main.info("--no_clean is active (or implicit due to --work_dir). Retaining work directory.")
        except Exception as e:
            L_main.error(f"An error occurred in main_wf with specified work_dir: {e}", exc_info=True)
            runlog["warnings"].append(f"ERROR: {str(e)}")
            success = False
    else:
        temp_dir_context = temp_dir(tag=f"{script_name_stem}_work", keep=args.no_clean, base_dir=str(args.output_dir))
        try:
            with temp_dir_context as temp_d_str:
                temp_work_dir_path = Path(temp_d_str)
                L_main.info(f"Using temporary work directory: {temp_work_dir_path}")
                runlog["steps"].append(f"Using temporary work directory: {temp_work_dir_path}")
                success = main_wf(args, L_main, temp_work_dir_path, final_stl_output_dir, runlog)
                if args.no_clean: 
                     runlog["warnings"].append(f"Temporary work directory retained: {temp_work_dir_path}")
                     L_main.warning(f"Temporary work directory retained by --no_clean: {temp_work_dir_path}")
        except Exception as e: 
            L_main.error(f"An error occurred during temporary directory handling or main workflow: {e}", exc_info=True)
            runlog["warnings"].append(f"ERROR: {str(e)}")
            success = False 
                
    if "args" in runlog: 
        for key, value in runlog["args"].items():
            if isinstance(value, Path): runlog["args"][key] = str(value)
                
    write_log(runlog, str(final_stl_output_dir), base_name=f"{script_name_stem}_log")

    if success: L_main.info("Multi-material slab component generation finished successfully.")
    else: L_main.error("Multi-material slab component generation failed. See logs for details."); sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# brain_for_printing/cli_multi_material_slab_components.py

"""
Generates distinct, non-overlapping multi-material brain slab components
suitable for 3D printing. Workflow:
1. Initial AC-PC Alignment of Input T1w Image.
2. Create High-Resolution AC-PC Master Template.
3. Generate Parent Surfaces in Native T1w Space.
4. Voxelize Parent Surfaces onto AC-PC Master Template (transforming them to AC-PC space).
5. Determine Global Bounding Box & Crop AC-PC Volumes.
6. Volumetric Slicing in AC-PC Space.
7. Define Material Volumes per Slab (AC-PC Space).
8. Mesh AC-PC Material Slabs using voxel2mesh (MRtrix3) to create STL files.

The final STLs are saved in AC-PC space in a specified output directory.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import shutil
import uuid

import nibabel as nib
import numpy as np
from typing import Dict, Optional, List, Any, Tuple

# --- Local Imports ---
from .io_utils import temp_dir, require_cmds, flexible_match, run_cmd
from .log_utils import get_logger, write_log
from .surfgen_utils import generate_single_brain_mask_surface, generate_brain_surfaces
from .five_tt_utils import run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes, is_vtk_available
# mesh_utils.volume_to_gifti and gifti_to_trimesh might not be needed if vol_to_mesh is fully replaced
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
                     "Accurate transformations (e.g. for Step 4 voxelization) might be impacted if only FSL .mat files are used without fslpy interpretation.")
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

def acpc_align_t1w(
    input_t1w_path: Path,
    output_acpc_t1w_path: Path,
    mni_template_path: Path,
    temp_dir_path: Path, 
    logger: logging.Logger,
    verbose: bool = False
) -> Tuple[bool, Optional[Path], Optional[Path]]:
    logger.info(f"Starting AC-PC alignment for {input_t1w_path.name} -> {output_acpc_t1w_path.name}")
    require_cmds(["flirt", "robustfov"], logger=logger)

    if not mni_template_path.exists():
        logger.error(f"MNI template not found at {mni_template_path}.")
        return False, None, None

    robustfov_out_path = temp_dir_path / f"{input_t1w_path.name.replace('.nii.gz','').replace('.nii','')}_robustfov.nii.gz"
    t1w_to_flirt_actual: Path = input_t1w_path

    try:
        logger.debug(f"Running robustfov command: {' '.join(['robustfov', '-i', str(input_t1w_path), '-r', str(robustfov_out_path)])}")
        run_cmd(["robustfov", "-i", str(input_t1w_path), "-r", str(robustfov_out_path)], verbose=verbose)
        if robustfov_out_path.exists() and robustfov_out_path.stat().st_size > 0:
            t1w_to_flirt_actual = robustfov_out_path
            logger.info(f"robustfov completed, using {t1w_to_flirt_actual.name} for FLIRT to AC-PC.")
        else:
            logger.warning(f"robustfov output '{robustfov_out_path.name}' not created or empty. "
                           f"Using original T1w '{input_t1w_path.name}' for FLIRT to AC-PC.")
    except Exception as e_fov:
        logger.warning(f"robustfov failed: {e_fov}. Using original T1w '{input_t1w_path.name}' for FLIRT to AC-PC.", exc_info=verbose)

    flirt_output_mat_path = temp_dir_path / f"{t1w_to_flirt_actual.name.replace('.nii.gz','').replace('.nii','')}_to_acpc.mat"

    flirt_cmd = [
        "flirt", "-in", str(t1w_to_flirt_actual), "-ref", str(mni_template_path),
        "-out", str(output_acpc_t1w_path), "-omat", str(flirt_output_mat_path),
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
            return True, t1w_to_flirt_actual, flirt_output_mat_path
        else:
            logger.error(f"FLIRT ran but output AC-PC aligned T1w or transform ({flirt_output_mat_path.name}) not created/empty.")
            return False, t1w_to_flirt_actual, None
    except Exception as e:
        logger.error(f"AC-PC alignment using FLIRT failed: {e}", exc_info=verbose)
        return False, t1w_to_flirt_actual, None

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

def run_voxel2mesh(
    input_nifti_path: Path,
    output_stl_path: Path,
    logger: logging.Logger,
    verbose: bool = False,
    threshold_value: Optional[float] = 0.5, 
    blocky: bool = False
) -> bool:
    logger.info(f"Running voxel2mesh: {input_nifti_path.name} -> {output_stl_path.name}")
    cmd = ["voxel2mesh"]
    if blocky:
        cmd.append("-blocky")
    if threshold_value is not None and not blocky:
        cmd.extend(["-threshold", str(threshold_value)])
    
    cmd.extend(["-force"]) # Overwrite output if it exists
    cmd.extend([str(input_nifti_path), str(output_stl_path)])

    try:
        output_stl_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        run_cmd(cmd, verbose=verbose)
        if output_stl_path.exists() and output_stl_path.stat().st_size > 0:
            logger.info(f"Successfully created mesh with voxel2mesh: {output_stl_path.name}")
            return True
        else:
            logger.error(f"voxel2mesh ran but output STL {output_stl_path.name} not created or is empty.")
            try: # Check if input was empty
                img_check = nib.load(str(input_nifti_path))
                if np.sum(img_check.get_fdata()) == 0:
                    logger.warning(f"Input NIfTI {input_nifti_path.name} to voxel2mesh was empty.")
            except Exception: pass
            return False
    except Exception as e:
        logger.error(f"voxel2mesh command failed for {input_nifti_path.name}: {e}", exc_info=verbose)
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
    parser.add_argument("--output_dir", default="./multi_material_slabs_acpc_stls", # Changed default to reflect new output space
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
                        help="Threshold for binarizing partial volume images from mesh2voxel (Step 4).")
    parser.add_argument("--voxel2mesh_threshold", type=float, default=0.5,
                        help="Threshold for voxel2mesh when generating final STLs from AC-PC material volumes (Step 8).")
    parser.add_argument("--voxel2mesh_blocky", action="store_true",
                        help="Use the -blocky option for voxel2mesh, creating voxel-face meshes instead of smooth marching cubes.")
    parser.add_argument("--skip_outer_csf", action="store_true", help="Skip generation of the outer CSF component.")
    parser.add_argument("--no_clean", action="store_true", help="Keep work directory if it was temporary.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")
    return parser


def main_wf(args: argparse.Namespace, L: logging.Logger, work_dir: Path, stl_output_base_dir: Path, runlog: Dict[str, Any]):
    runlog["steps"].append("Starting multi-material slab component generation.")
    L.info(f"Target AC-PC aligned voxel resolution: {args.voxel_resolution} mm")
    L.info(f"Slab thickness: {args.slab_thickness}mm, Orientation (AC-PC): {args.slab_orientation}")
    L.info(f"BrainMask inflation: {args.brain_mask_inflate_mm}mm")
    L.info(f"voxel2mesh settings: threshold={args.voxel2mesh_threshold}, blocky={args.voxel2mesh_blocky}")
    L.info("Final slab STLs will be generated in AC-PC space.")

    mrtrix_cmds = ["mrgrid", "mesh2voxel", "voxel2mesh"] # Added voxel2mesh
    fsl_cmds = ["flirt", "fslroi", "robustfov", "convert_xfm"]
    # ANTs no longer strictly required if resampling step is gone, but good to keep for other tools
    ants_cmds = ["antsApplyTransforms"] 
    require_cmds(mrtrix_cmds + fsl_cmds + ants_cmds, logger=L)

    # Define directories
    acpc_align_dir = work_dir / "00_acpc_alignment"
    parent_surf_gen_dir = work_dir / "01_parent_surface_generation"
    full_vol_voxelized_dir = work_dir / "02_full_acpc_voxelized_pvs"
    full_vol_binarized_dir = work_dir / "03_full_acpc_binarized_masks"
    cropped_full_vol_dir = work_dir / "04_cropped_full_acpc_masks"
    volumetric_slabs_acpc_dir = work_dir / "05_volumetric_slabs_acpc_nifti"
    material_slabs_acpc_vol_dir = work_dir / "06_material_slabs_acpc_volumes_nifti"
    # Removed: material_slabs_native_vol_dir
    # Removed: temp_ants_xfm_dir
    
    dirs_to_create = [acpc_align_dir, parent_surf_gen_dir, full_vol_voxelized_dir,
                   full_vol_binarized_dir, cropped_full_vol_dir, volumetric_slabs_acpc_dir,
                   material_slabs_acpc_vol_dir] # Removed native and ants_temp dirs
        
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
    
    acpc_success, actual_flirt_src_path, t1w_to_acpc_fsl_mat_path = acpc_align_t1w(
        original_t1w_path_for_ref,
        t1w_acpc_aligned_path,
        mni_template_for_acpc,
        acpc_align_dir, L, args.verbose
    )
    if not acpc_success or actual_flirt_src_path is None or t1w_to_acpc_fsl_mat_path is None:
        L.error("AC-PC alignment step failed."); return False
    
    runlog["steps"].append(f"AC-PC T1w: {t1w_acpc_aligned_path.name}, Native-to-ACPC FSL XFM: {t1w_to_acpc_fsl_mat_path.name}")

    # --- Derive World-to-World Affines ---
    # native_to_acpc_world_affine is needed for Step 4
    native_to_acpc_world_affine: Optional[np.ndarray] = None
    if FSLPY_AVAILABLE: 
        L.info("Attempting to derive native-to-ACPC world-to-world affine using fslpy...")
        native_to_acpc_world_affine = get_flirt_world_to_world_affine(
            fsl_flirt_mat_path=t1w_to_acpc_fsl_mat_path,
            src_image_path=actual_flirt_src_path, 
            ref_image_path=mni_template_for_acpc, # MNI template is the reference for AC-PC space
            logger=L
        )
        if native_to_acpc_world_affine is not None:
            L.info("Successfully derived native-to-ACPC world affine.")
        else:
            L.error("Failed to derive native-to-ACPC world affine using fslpy. Voxelization in Step 4 might be misaligned.")
            runlog["warnings"].append("Failed to derive native-to-ACPC world affine. Potential misalignment in Step 4.")
    else: 
        L.warning("fslpy is not available. Native-to-ACPC world-to-world affine cannot be reliably computed from FSL .mat. "
                  "Step 4 (voxelization) may be misaligned if relying on this transform.")
        runlog["warnings"].append("fslpy not available. native_to_acpc_world_affine might be inaccurate. Potential Step 4 misalignment.")


    # --- Step 2: Creating High-Resolution Master AC-PC Template ---
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
    L.info("--- Step 3: Generating full parent T1-space surfaces (Trimesh objects) ---")
    # (This section remains largely the same as it generates native T1w Trimesh objects)
    parent_meshes_trimesh: Dict[str, Optional[Any]] = {} # Changed to Any for trimesh
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
        elif item == "cerebellum": white_other_explicit_for_white_complex.add("cerebellum_wm") 
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
    L.info(f"--- Step 4: Voxelizing full parent surfaces onto Master AC-PC Template ({master_hires_acpc_template_path.name}) ---")
    full_acpc_binarized_masks: Dict[str, Path] = {}
    temp_obj_export_dir_for_acpc_voxelization = full_vol_voxelized_dir / "temp_acpc_aligned_objs"
    temp_obj_export_dir_for_acpc_voxelization.mkdir(parents=True, exist_ok=True)

    for parent_name, trimesh_native_obj_any in valid_parent_meshes_trimesh.items():
        if trimesh_native_obj_any is None: continue 
        trimesh_native_obj = trimesh_native_obj_any # Cast to trimesh.Trimesh if needed, assuming it is

        safe_name = parent_name.replace("_", "-").replace(" ", "-")
        temp_acpc_aligned_surf_obj_path = temp_obj_export_dir_for_acpc_voxelization / f"{args.subject_id}_desc-{safe_name}_full_surf_acpc-aligned.obj"
        
        # Explicitly use trimesh type for copy and transform
        if not hasattr(trimesh_native_obj, 'copy') or not hasattr(trimesh_native_obj, 'apply_transform'):
            L.error(f"Object for parent_name '{parent_name}' is not a Trimesh object. Skipping voxelization.")
            continue
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
    L.info("--- Step 5: Determining global bounding box for cropping full AC-PC volumes ---")
    # (This section remains the same)
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

    # --- Step 6: Performing volumetric slicing (in AC-PC space) ---
    L.info("--- Step 6: Performing volumetric slicing (in AC-PC space) ---")
    # (This section remains the same)
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
    
    total_extent_cropped_vox = cropped_vol_dims_xyz[slice_axis_idx]

    while current_slab_start_voxel_in_cropped < total_extent_cropped_vox:
        slab_idx = num_slabs_generated
        volumetric_slabs_acpc_struct_files[slab_idx] = {}
        
        actual_slab_thickness_this_iteration_vox = min(slab_thickness_voxels, total_extent_cropped_vox - current_slab_start_voxel_in_cropped)
        if actual_slab_thickness_this_iteration_vox <= 0: break 

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
        # (This section remains largely the same, it defines AC-PC NIfTI material volumes)
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

        # This dictionary will hold the paths to the final AC-PC material NIfTIs for this slab
        final_material_volumes_for_meshing: Dict[str, Path] = {} 

        if not args.skip_outer_csf:
            vol_outer_csf_acpc = vol_subtract_numpy(m_bm_acpc, m_pc_acpc)
            p_acpc = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-OuterCSF_acpc.nii.gz"
            if save_numpy_as_nifti(vol_outer_csf_acpc, acpc_slab_template_img_for_saving, p_acpc, L): 
                final_material_volumes_for_meshing["OuterCSF"] = p_acpc
        
        vol_gm_acpc = vol_subtract_numpy(m_pc_acpc, m_wc_acpc)
        p_acpc = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-GreyMatter_acpc.nii.gz"
        if save_numpy_as_nifti(vol_gm_acpc, acpc_slab_template_img_for_saving, p_acpc, L): 
            final_material_volumes_for_meshing["GreyMatter"] = p_acpc

        vol_vent_final_acpc = m_vent_acpc.copy() if m_vent_acpc is not None else zeros_for_fallback_acpc_slab.copy()
        p_acpc_vent = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-Ventricles_acpc.nii.gz"
        if save_numpy_as_nifti(vol_vent_final_acpc, acpc_slab_template_img_for_saving, p_acpc_vent, L): 
            final_material_volumes_for_meshing["Ventricles"] = p_acpc_vent

        vol_sgm_final_acpc = m_sgm_acpc.copy() if m_sgm_acpc is not None else zeros_for_fallback_acpc_slab.copy()
        p_acpc_sgm = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-SubcorticalGrey_acpc.nii.gz"
        if save_numpy_as_nifti(vol_sgm_final_acpc, acpc_slab_template_img_for_saving, p_acpc_sgm, L): 
            final_material_volumes_for_meshing["SubcorticalGrey"] = p_acpc_sgm
        
        working_wm_acpc = m_wc_acpc.copy() if m_wc_acpc is not None else zeros_for_fallback_acpc_slab.copy()
        working_wm_acpc = vol_subtract_numpy(working_wm_acpc, vol_vent_final_acpc)
        working_wm_acpc = vol_subtract_numpy(working_wm_acpc, vol_sgm_final_acpc)
        vol_wm_final_acpc = working_wm_acpc
        p_acpc_wm = material_slabs_acpc_vol_dir / f"{args.subject_id}_slab-{slab_idx}_desc-WhiteMatter_acpc.nii.gz"
        if save_numpy_as_nifti(vol_wm_final_acpc, acpc_slab_template_img_for_saving, p_acpc_wm, L): 
            final_material_volumes_for_meshing["WhiteMatter"] = p_acpc_wm
        
        runlog["steps"].append(f"Slab {slab_idx}: Volumetric material definition in AC-PC space completed.")


        # --- NEW Step 8: Meshing AC-PC Material Slabs using voxel2mesh ---
        L.info(f"--- Slab {slab_idx}: Meshing AC-PC material slab volumes using voxel2mesh ---")
        
        material_vol_names_for_mesh = ["OuterCSF", "GreyMatter", "WhiteMatter", "Ventricles", "SubcorticalGrey"]
        if args.skip_outer_csf and "OuterCSF" in material_vol_names_for_mesh: 
            material_vol_names_for_mesh.remove("OuterCSF")

        if not final_material_volumes_for_meshing:
            L.warning(f"Slab {slab_idx}: No AC-PC material volumes were defined or saved. Skipping meshing for this slab.")
            continue

        for mat_name in material_vol_names_for_mesh:
            acpc_vol_path_for_mesh = final_material_volumes_for_meshing.get(mat_name) 
            if acpc_vol_path_for_mesh and acpc_vol_path_for_mesh.exists():
                # Output STLs will be in AC-PC space
                stl_path = stl_output_base_dir / f"{args.subject_id}_slab-{slab_idx}_desc-{mat_name}_material_space-acpc.stl"
                
                mesh_generated_successfully = run_voxel2mesh(
                    input_nifti_path=acpc_vol_path_for_mesh,
                    output_stl_path=stl_path,
                    logger=L,
                    verbose=args.verbose,
                    threshold_value=args.voxel2mesh_threshold,
                    blocky=args.voxel2mesh_blocky
                )

                if mesh_generated_successfully and stl_path.exists() and stl_path.stat().st_size > 0:
                    runlog["output_files"].append(str(stl_path))
                    L.info(f"Successfully generated STL for {mat_name}, slab {slab_idx}: {stl_path.name}")
                elif mesh_generated_successfully: # implies stl_path check failed
                     L.warning(f"voxel2mesh reported success for {mat_name} slab {slab_idx} but STL file {stl_path.name} is missing or empty.")
                     runlog["warnings"].append(f"voxel2mesh success but STL missing/empty: {stl_path.name}")
                else: # mesh_generated_successfully is False
                    L.warning(f"Failed to generate mesh for {mat_name}, slab {slab_idx} from {acpc_vol_path_for_mesh.name}.")
                    runlog["warnings"].append(f"voxel2mesh failed for {mat_name} slab {slab_idx} ({acpc_vol_path_for_mesh.name})")
            else:
                L.debug(f"AC-PC volume for {mat_name} slab {slab_idx} not found in 'final_material_volumes_for_meshing' or does not exist. Skipping mesh generation.")
        
        runlog["steps"].append(f"Slab {slab_idx}: Meshing of AC-PC material volumes with voxel2mesh completed.")
    
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
            else: # Clean work_dir if not --no_clean and work_dir was specified
                if work_dir_path.exists(): 
                    L_main.info(f"Cleaning specified work directory: {work_dir_path}")
                    try: shutil.rmtree(work_dir_path)
                    except Exception as e_clean_spec: L_main.warning(f"Could not clean specified work_dir {work_dir_path}: {e_clean_spec}")
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

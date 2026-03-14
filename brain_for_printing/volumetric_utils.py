# brain_for_printing/volumetric_utils.py
"""
Utilities for volumetric operations, including wrapping MRtrix commands
like mrgrid, mesh2voxel, and performing numpy-based volume math.
"""
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Optional, List, Dict, Any # Ensure all typing imports are here

import nibabel as nib
import numpy as np
import trimesh # trimesh might not be directly used here but good for context

from .io_utils import run_cmd 

L = logging.getLogger(__name__)

def regrid_to_resolution(
    input_nifti_path: Path,
    output_nifti_path: Path,
    voxel_size_mm: float,
    logger: logging.Logger, 
    verbose: bool = False
) -> bool:
    """
    Regrids an input NIfTI image to a specified isotropic voxel resolution using mrgrid.
    """
    logger.info(f"Regridding {input_nifti_path.name} to {voxel_size_mm}mm isotropic resolution -> {output_nifti_path.name}")
    cmd = [
        "mrgrid", str(input_nifti_path), "regrid",
        "-voxel", str(voxel_size_mm),
        str(output_nifti_path),
        "-force" 
    ]
    try:
        run_cmd(cmd, verbose=verbose)
        if output_nifti_path.exists():
            logger.info("Regridding successful.")
            return True
        else:
            logger.error("mrgrid command ran but output file was not created.")
            return False
    except Exception as e:
        logger.error(f"mrgrid command failed: {e}")
        return False

def mesh_to_partial_volume(
    input_mesh_path: Path,
    template_nifti_path: Path,
    output_pv_nifti_path: Path,
    logger: logging.Logger, 
    verbose: bool = False
) -> bool:
    """
    Converts a mesh surface to a partial volume estimation NIfTI image using mesh2voxel.
    """
    logger.info(f"Voxelizing {input_mesh_path.name} using template {template_nifti_path.name} -> {output_pv_nifti_path.name}")
    cmd = [
        "mesh2voxel", str(input_mesh_path), str(template_nifti_path), str(output_pv_nifti_path),
        "-force" 
    ]
    try:
        run_cmd(cmd, verbose=verbose)
        if output_pv_nifti_path.exists():
            logger.info("mesh2voxel successful.")
            return True
        else:
            logger.error("mesh2voxel command ran but output file was not created.")
            return False
    except Exception as e:
        logger.error(f"mesh2voxel command failed: {e}")
        return False

def binarize_volume_file(
    input_pv_nifti_path: Path,
    output_binary_nifti_path: Path,
    threshold: float = 0.5,
    logger: Optional[logging.Logger] = None 
) -> bool:
    """Loads a partial volume NIfTI, binarizes it based on a threshold, and saves it."""
    local_logger = logger or L 
    local_logger.debug(f"Binarizing {input_pv_nifti_path.name} with threshold {threshold} -> {output_binary_nifti_path.name}")
    try:
        img = nib.load(str(input_pv_nifti_path))
        data = img.get_fdata()
        binary_data = (data > threshold).astype(np.uint8)
        
        binary_img = nib.Nifti1Image(binary_data, img.affine, img.header)
        output_binary_nifti_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        nib.save(binary_img, str(output_binary_nifti_path))
        local_logger.info(f"Saved binarized volume: {output_binary_nifti_path.name}")
        return True
    except Exception as e:
        local_logger.error(f"Failed to binarize volume {input_pv_nifti_path.name}: {e}")
        return False

def save_numpy_as_nifti(
    data_array: np.ndarray,
    template_nifti_image: nib.Nifti1Image, 
    output_path: Path,
    logger: logging.Logger, 
    datatype: str = 'uint8' 
) -> bool:
    """Saves a NumPy array as a NIfTI file, using affine and header from a template."""
    logger.debug(f"Saving numpy array to NIfTI: {output_path.name}")
    try:
        processed_data = data_array.astype(getattr(np, datatype))
        new_img = nib.Nifti1Image(processed_data, template_nifti_image.affine, template_nifti_image.header)
        output_path.parent.mkdir(parents=True, exist_ok=True) 
        nib.save(new_img, str(output_path))
        logger.info(f"Saved NIfTI: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save NIfTI {output_path.name}: {e}", exc_info=True)
        return False

def vol_subtract_numpy(vol_a_data: np.ndarray, vol_b_data: np.ndarray) -> np.ndarray:
    """Computes (A AND NOT B). Result is binary."""
    return np.logical_and(vol_a_data, np.logical_not(vol_b_data)).astype(vol_a_data.dtype)

def vol_intersect_numpy(vol_a_data: np.ndarray, vol_b_data: np.ndarray) -> np.ndarray:
    """Computes (A AND B). Result is binary."""
    return np.logical_and(vol_a_data, vol_b_data).astype(vol_a_data.dtype)

def vol_union_numpy(vol_list: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Computes the union (OR) of a list of binary volumes. Skips None entries."""
    valid_vols = [v for v in vol_list if v is not None]
    if not valid_vols:
        return None 
    
    base_vol = valid_vols[0]
    if len(valid_vols) == 1:
        union_result = base_vol.copy() # Important to copy if only one
    else:
        union_result = np.zeros_like(base_vol, dtype=base_vol.dtype)
        for vol_data in valid_vols:
            if vol_data.shape != union_result.shape:
                L.error(f"Volume shape mismatch in union: expected {union_result.shape}, got {vol_data.shape}")
                continue 
            union_result = np.logical_or(union_result, vol_data)
        
    return union_result.astype(base_vol.dtype)


def load_nifti_data(nifti_path: Path, logger: logging.Logger) -> Optional[np.ndarray]: 
    """Loads NIfTI data as a numpy array, assumed to be uint8 for binary masks."""
    if not nifti_path.exists():
        logger.warning(f"NIfTI file not found for loading: {nifti_path}")
        return None
    try:
        img = nib.load(str(nifti_path))
        data = img.get_fdata()
        # If these are binary masks from binarize_volume_file, they are already uint8.
        # If they could be other types (e.g. float PV maps), this cast might be lossy.
        # For the purpose of boolean ops on masks, uint8 (0 or 1) is fine.
        return data.astype(np.uint8) 
    except Exception as e:
        logger.error(f"Failed to load NIfTI data from {nifti_path}: {e}")
        return None

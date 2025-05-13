"""
aseg_utils.py
------------
Utilities for working with ASEG (Automatic Segmentation) files.
Includes functions for extracting structure masks and generating surfaces.
"""

import logging
from pathlib import Path
from typing import List, Optional
import uuid
import subprocess

from .io_utils import run_cmd, flexible_match
from .mesh_utils import volume_to_gifti

L = logging.getLogger(__name__)

def _extract_structure_mask_t1(
    aseg_file: str,
    structure: str,
    output_file: Path,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Extract binary mask for a structure from ASEG file.

    Args:
        aseg_file: Path to ASEG file
        structure: Structure name (e.g., 'brainstem', 'cerebellum')
        output_file: Path to output mask file
        verbose: Enable verbose logging
        logger: Logger instance (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    # Map structure names to label IDs
    structure_labels = {
        'brainstem': [16],  # Brain Stem
        'cerebellum': [7, 8, 46, 47],  # Left/Right Cerebellum Cortex
        'corpus_callosum': [251, 252, 253, 254, 255],  # Corpus Callosum
    }
    
    if structure not in structure_labels:
        logger.error(f"Unknown structure: {structure}")
        return False
        
    label_ids = structure_labels[structure]
    
    try:
        # Use mri_binarize to extract mask
        cmd = [
            "mri_binarize",
            "--i", aseg_file,
            "--o", str(output_file),
            "--match", *[str(label) for label in label_ids],
        ]
        run_cmd(cmd, verbose=verbose)
        return True
    except Exception as e:
        logger.error(f"Failed to extract mask: {str(e)}")
        return False

def _generate_surface_from_mask(
    mask_file: Path,
    output_file: Path,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Generate surface from binary mask.

    Args:
        mask_file: Path to binary mask file
        output_file: Path to output GIFTI surface file
        verbose: Enable verbose logging
        logger: Logger instance (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    try:
        # Convert NIfTI to GIFTI using marching cubes
        volume_to_gifti(str(mask_file), str(output_file), level=0.5)
        logger.info(f"Created GIFTI: {output_file.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate surface: {str(e)}")
        return False

def extract_structure_surface(
    structure: str,
    output_dir: str,
    # --- Inputs ---
    aseg_file_path: Optional[str] = None, # Input: Path to the specific ASEG file
    subject_id: Optional[str] = None,     # Input: Used for output naming convention
    target_space: str = "T1",             # Input: Used for output naming convention
    # --- Options ---
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    # --- Removed inputs no longer needed for search ---
    # session: Optional[str] = None,
    # run: Optional[str] = None,
    # subjects_dir: Optional[str] = None,
) -> Optional[Path]:
    """Extracts a surface mesh for a specified anatomical structure from a given ASEG segmentation file.

    Args:
        structure: Name of the structure to extract (e.g., 'brainstem').
        output_dir: Directory where intermediate mask and final GIFTI file will be saved.
        aseg_file_path: Optional. The specific ASEG file (.nii.gz or .mgz) to use.
                        If None, the function will log an error and return None.
        subject_id: Optional. Subject ID (e.g., 'sub-01'). Used for naming output files.
        target_space: Optional. String indicating the space (e.g., 'T1'). Used for naming.
        verbose: Enable detailed logging.
        logger: Optional logger instance.

    Returns:
        Optional[Path]: Path object to the generated GIFTI surface file if successful, None otherwise.
    """
    logger = logger or L # Use provided logger or module logger
    logger.info(f"--- Starting surface extraction for structure: '{structure}' ---")

    # --- Check if aseg_file_path is provided ---
    if not aseg_file_path:
        logger.error("ASEG file path was not provided to extract_structure_surface. Cannot proceed.")
        return None

    aseg_in_target_space = Path(aseg_file_path) # Convert to Path object
    logger.info(f"Using provided ASEG file: {aseg_in_target_space}")
    if not aseg_in_target_space.exists():
         logger.error(f"Provided ASEG file not found: {aseg_in_target_space}")
         return None
    # --- End Check ---


    # --- Create output directory ---
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir_path}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return None


    # --- Define output filenames ---
    # Use a unique ID to prevent clashes if called multiple times quickly
    unique_id = uuid.uuid4().hex[:6]
    # Base filename includes subject ID if available, otherwise just structure/space
    fname_prefix = f"{subject_id}_" if subject_id else ""
    # Use the helper function names exactly as defined in your file
    structure_mask_file = output_dir_path / f"{fname_prefix}space-{target_space}_desc-{structure}_mask_{unique_id}.nii.gz"
    final_surface_file = output_dir_path / f"{fname_prefix}space-{target_space}_desc-{structure}_surface_{unique_id}.gii"


    # --- Step 1: Generate binary mask for the structure ---
    logger.info(f"Step 1: Extracting binary mask for '{structure}'...")
    # Use the validated aseg_in_target_space path
    # Ensure the helper function name matches your file (_extract_structure_mask_t1 or _extract_structure_mask_from_file)
    mask_ok = _extract_structure_mask_t1(
        aseg_file=str(aseg_in_target_space), # Pass the validated path string
        structure=structure,
        output_file=structure_mask_file,
        verbose=verbose,
        logger=logger,
    )

    if not mask_ok:
        logger.error(f"Failed to create binary mask for '{structure}'. Aborting extraction.")
        structure_mask_file.unlink(missing_ok=True) # Attempt cleanup
        return None


    # --- Step 2: Generate surface from the mask ---
    logger.info(f"Step 2: Generating GIFTI surface from mask '{structure_mask_file.name}'...")
    surface_ok = _generate_surface_from_mask(
        mask_file=structure_mask_file,
        output_file=final_surface_file,
        verbose=verbose, # Pass verbose along
        logger=logger,
    )

    # --- Cleanup intermediate mask file ---
    logger.debug(f"Cleaning up intermediate mask file: {structure_mask_file.name}")
    structure_mask_file.unlink(missing_ok=True) # Delete mask regardless of surface success

    if not surface_ok:
        logger.error(f"Failed to generate surface for '{structure}'.")
        final_surface_file.unlink(missing_ok=True) # Attempt cleanup
        return None

    logger.info(f"--- Successfully extracted surface for '{structure}' => {final_surface_file.name} ---")
    return final_surface_file


def convert_fs_aseg_to_t1w(
    subjects_dir: str,
    subject_id: str,
    output_dir: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    verbose: bool = False
) -> Optional[str]:
    """
    Convert FreeSurfer ASEG file to T1w space.
    
    Args:
        subjects_dir: Path to subjects directory
        subject_id: Subject ID (e.g., 'sub-01')
        output_dir: Directory to save the converted ASEG file
        session: BIDS session ID
        run: BIDS run ID
        verbose: Enable verbose logging
        
    Returns:
        Optional[str]: Path to the converted ASEG file in T1w space, None if failed
    """
    try:
        # Get paths
        subject_id_clean = subject_id.replace('sub-', '')
        fs_subject_dir = Path(subjects_dir) / "sourcedata" / "freesurfer" / f"sub-{subject_id_clean}"
        anat_dir = Path(subjects_dir) / f"sub-{subject_id_clean}" / "anat"
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Find FreeSurfer ASEG file
        aseg_fs = fs_subject_dir / "mri" / "aseg.mgz"
        if not aseg_fs.exists():
            L.error(f"FreeSurfer ASEG not found: {aseg_fs}")
            return None
            
        # Find transform file
        xfm_path = flexible_match(
            anat_dir,
            subject_id,
            descriptor="from-fsnative_to-T1w_mode-image",
            suffix="xfm",
            ext=".txt",
            session=session,
            run=run,
            logger=L
        )
        
        # Find T1w reference
        t1w_ref = flexible_match(
            anat_dir,
            subject_id,
            descriptor="preproc",
            suffix="T1w",
            ext=".nii.gz",
            session=session,
            run=run,
            logger=L
        )
        
        # Convert ASEG to NIfTI
        aseg_nii = output_dir_path / f"{subject_id}_desc-aseg_fsnative.nii.gz"
        run_cmd([
            "mri_convert",
            str(aseg_fs),
            str(aseg_nii)
        ], verbose=verbose)
        
        # Apply transform to get into T1w space
        output_filename = f"{subject_id}"
        if session:
            output_filename = f"{output_filename}_ses-{session}"
        output_filename = f"{output_filename}_desc-aseg_dseg.nii.gz"
        aseg_t1w = output_dir_path / output_filename
            
        run_cmd([
            "antsApplyTransforms",
            "-d", "3",
            "-i", str(aseg_nii),
            "-r", t1w_ref,
            "-t", xfm_path,
            "-o", str(aseg_t1w),
            "-n", "NearestNeighbor"
        ], verbose=verbose)
        
        # Clean up intermediate file
        aseg_nii.unlink()
        
        return str(aseg_t1w)
        
    except FileNotFoundError as e:
        L.error(f"File not found: {e}")
        return None
    except Exception as e:
        L.error(f"Failed to convert ASEG: {e}", exc_info=verbose)
        return None 

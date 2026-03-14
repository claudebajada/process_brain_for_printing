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
# import subprocess # Not directly used in _extract_structure_mask_t1 if run_cmd is used

from .io_utils import run_cmd, flexible_match # flexible_match might not be needed here
from .mesh_utils import volume_to_gifti
from . import constants as const # Import constants to access label definitions

L = logging.getLogger(__name__)

# Define a more comprehensive map at module level or ensure it's used in _extract_structure_mask_t1
ASEG_STRUCTURE_TO_LABELS = {
    "brainstem": const.BRAINSTEM_LABEL,
    "cerebellum_cortex": const.CEREBELLUM_CORTEX_LABELS,
    "cerebellum_wm": const.CEREBELLUM_WM_LABELS,
    "cerebellum": const.CEREBELLUM_LABELS, # Combined cerebellum
    "corpus_callosum": const.CORPUS_CALLOSUM_LABELS,
}

def _extract_structure_mask_t1(
    aseg_file: str,
    structure: str, # This 'structure' name must be a key in ASEG_STRUCTURE_TO_LABELS
    output_file: Path,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Extract binary mask for a structure from ASEG file."""
    logger = logger or L # Use provided or module logger
    
    if structure not in ASEG_STRUCTURE_TO_LABELS:
        logger.error(f"Unknown structure for ASEG mask extraction: '{structure}'. Known structures are: {list(ASEG_STRUCTURE_TO_LABELS.keys())}")
        return False
        
    label_ids = ASEG_STRUCTURE_TO_LABELS[structure]
    if not label_ids: # Should not happen if constants are defined
        logger.error(f"Label IDs for structure '{structure}' are empty or undefined. Cannot extract mask.")
        return False
    
    logger.debug(f"Extracting mask for '{structure}' using labels {label_ids} from {Path(aseg_file).name} into {output_file.name}")
    
    try:
        cmd = [
            "mri_binarize",
            "--i", aseg_file,
            "--o", str(output_file),
            "--match", *[str(label) for label in label_ids],
        ]
        run_cmd(cmd, verbose=verbose) # run_cmd should raise an error if mri_binarize fails
        if not output_file.exists() or output_file.stat().st_size == 0:
            logger.error(f"mri_binarize command ran for '{structure}' but output file {output_file} was not created or is empty.")
            return False
        logger.info(f"Successfully created binary mask for '{structure}': {output_file.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract mask for '{structure}': {e}", exc_info=verbose)
        return False

def _generate_surface_from_mask(
    mask_file: Path,
    output_file: Path,
    verbose: bool = False, # verbose not directly used by volume_to_gifti but good to keep consistent
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Generate surface from binary mask."""
    logger = logger or L
    
    try:
        # volume_to_gifti logs its own success/failure related to saving
        volume_to_gifti(str(mask_file), str(output_file), level=0.5)
        if not output_file.exists() or output_file.stat().st_size == 0:
             logger.error(f"volume_to_gifti seemed to run for mask '{mask_file.name}' but output GIFTI {output_file} was not created or is empty.")
             return False
        # The info log "Created GIFTI" is now inside volume_to_gifti.
        return True
    except Exception as e:
        logger.error(f"Failed to generate surface from mask '{mask_file.name}': {e}", exc_info=verbose)
        return False

def extract_structure_surface(
    structure: str,
    output_dir: str,
    aseg_file_path: Optional[str] = None, 
    subject_id: Optional[str] = None,     
    target_space: str = "T1",             
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Extracts a surface mesh for a specified anatomical structure from a given ASEG segmentation file."""
    logger = logger or L 
    logger.info(f"--- Starting surface extraction for structure: '{structure}' ---")

    if not aseg_file_path:
        logger.error("ASEG file path was not provided to extract_structure_surface. Cannot proceed.")
        return None

    aseg_in_target_space = Path(aseg_file_path)
    logger.info(f"Using provided ASEG file: {aseg_in_target_space}")
    if not aseg_in_target_space.exists():
         logger.error(f"Provided ASEG file not found: {aseg_in_target_space}")
         return None

    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir_path}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return None

    unique_id = uuid.uuid4().hex[:6]
    fname_prefix = f"{subject_id}_" if subject_id else ""
    
    # Use a more descriptive name for the mask file if possible
    safe_structure_name = structure.replace("_", "-") # Basic sanitization for filename
    structure_mask_file = output_dir_path / f"{fname_prefix}space-{target_space}_desc-{safe_structure_name}_mask_{unique_id}.nii.gz"
    final_surface_file = output_dir_path / f"{fname_prefix}space-{target_space}_desc-{safe_structure_name}_surface_{unique_id}.gii"

    logger.info(f"Step 1: Extracting binary mask for '{structure}'...")
    mask_ok = _extract_structure_mask_t1(
        aseg_file=str(aseg_in_target_space), 
        structure=structure, # This 'structure' name must be a key in ASEG_STRUCTURE_TO_LABELS
        output_file=structure_mask_file,
        verbose=verbose,
        logger=logger,
    )

    if not mask_ok:
        logger.error(f"Failed to create binary mask for '{structure}'. Aborting extraction.")
        structure_mask_file.unlink(missing_ok=True) 
        return None

    logger.info(f"Step 2: Generating GIFTI surface from mask '{structure_mask_file.name}'...")
    surface_ok = _generate_surface_from_mask(
        mask_file=structure_mask_file,
        output_file=final_surface_file,
        verbose=verbose, 
        logger=logger,
    )

    logger.debug(f"Cleaning up intermediate mask file: {structure_mask_file.name}")
    structure_mask_file.unlink(missing_ok=True) 

    if not surface_ok:
        logger.error(f"Failed to generate surface for '{structure}'.")
        final_surface_file.unlink(missing_ok=True) 
        return None

    logger.info(f"--- Successfully extracted surface for '{structure}' => {final_surface_file.name} ---")
    return final_surface_file


def convert_fs_aseg_to_t1w(
    subjects_dir: str, # This should be the top-level BIDS derivatives dir
    subject_id: str,
    output_dir: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    verbose: bool = False
) -> Optional[str]:
    """
    Convert FreeSurfer ASEG file (aseg.mgz expected in sourcedata/freesurfer) to T1w space.
    """
    try:
        subject_id_clean = subject_id.replace('sub-', '')
        # Path to sourcedata/freesurfer within the BIDS derivatives subjects_dir
        fs_sourcedata_dir = Path(subjects_dir) / "sourcedata" / "freesurfer" / f"sub-{subject_id_clean}"
        # Path to subject's anat directory in BIDS derivatives
        bids_anat_dir = Path(subjects_dir) / subject_id / "anat"
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        aseg_fs_mgz = fs_sourcedata_dir / "mri" / "aseg.mgz"
        if not aseg_fs_mgz.exists():
            L.error(f"FreeSurfer aseg.mgz not found at expected location: {aseg_fs_mgz}")
            return None
            
        # Transform is from fsnative to T1w (usually found in subject's anat dir from fmriprep)
        xfm_fsnative_to_t1w_path = flexible_match(
            bids_anat_dir,
            subject_id, # Pass full subject_id like "sub-XYZ"
            descriptor="from-fsnative_to-T1w_mode-image", # Common descriptor from fmriprep
            suffix="xfm",
            ext=".txt", # Often .txt for LTA transforms usable by mri_vol2vol or ANTs
            session=session,
            run=run,
            logger=L
        )
        
        # T1w reference image (usually preprocessed T1w in subject's anat)
        t1w_ref_image = flexible_match(
            bids_anat_dir,
            subject_id,
            descriptor="preproc", # Or other appropriate descriptor for your reference T1w
            suffix="T1w",
            ext=".nii.gz",
            session=session,
            run=run,
            logger=L
        )
        
        # Intermediate NIfTI from aseg.mgz (in fsnative space)
        aseg_fsnative_nii = output_dir_path / f"{subject_id}_desc-aseg_fromFS_fsnative.nii.gz"
        run_cmd([
            "mri_convert", str(aseg_fs_mgz), str(aseg_fsnative_nii)
        ], verbose=verbose)
        
        # Final output: aseg in T1w space
        output_filename_t1w = f"{subject_id}"
        if session: output_filename_t1w += f"_ses-{session.replace('ses-','')}"
        if run: output_filename_t1w += f"_run-{run.replace('run-','')}" # Ensure run is cleaned if needed
        output_filename_t1w += "_space-T1w_desc-aseg_dseg.nii.gz" # BIDS compliant name
        
        aseg_final_t1w_path = output_dir_path / output_filename_t1w
            
        # Use antsApplyTransforms for robust application of .txt (ITK format) or .h5 transforms
        # If xfm_fsnative_to_t1w_path is an LTA, it might need conversion or mri_vol2vol
        # Assuming xfm_path is compatible with antsApplyTransforms (e.g. .h5 or .mat from ANTs)
        # If it's an LTA (.lta or .txt LTA), mri_vol2vol would be more direct:
        # mri_vol2vol --mov aseg_fsnative_nii --targ t1w_ref_image --lta xfm_fsnative_to_t1w_path --o aseg_final_t1w_path --nearest
        # For now, assuming ANTs compatible transform as per previous structure:
        run_cmd([
            "antsApplyTransforms",
            "-d", "3",
            "-i", str(aseg_fsnative_nii),
            "-r", str(t1w_ref_image),
            "-t", str(xfm_fsnative_to_t1w_path),
            "-o", str(aseg_final_t1w_path),
            "-n", "GenericLabel" # Use GenericLabel or MultiLabel for segmentations
        ], verbose=verbose)
        
        aseg_fsnative_nii.unlink(missing_ok=True) # Clean up intermediate
        
        if aseg_final_t1w_path.exists():
            return str(aseg_final_t1w_path)
        else:
            L.error(f"Failed to create ASEG in T1w space at {aseg_final_t1w_path}")
            return None
        
    except FileNotFoundError as e:
        L.error(f"File not found during ASEG to T1w conversion: {e}")
        return None
    except Exception as e:
        L.error(f"Failed to convert ASEG to T1w: {e}", exc_info=verbose)
        return None

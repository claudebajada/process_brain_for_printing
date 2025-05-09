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
    subject_id: str,
    structure: str,
    target_space: str = "T1",
    output_dir: Optional[str] = None,
    session: Optional[str] = None,
    run: Optional[str] = None,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Extract surface from ASEG structure in specified space.

    Args:
        subject_id: Subject ID
        structure: Structure name (e.g., 'brainstem', 'cerebellum')
        target_space: Target space ('T1' or 'MNI')
        output_dir: Output directory (default: current directory)
        session: Session ID (optional)
        run: Run ID (optional)
        verbose: Enable verbose logging
        logger: Logger instance (optional)

    Returns:
        Path to generated GIFTI file or None if failed
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Locating {target_space}-space aseg for {subject_id} ({structure})")

    # Try to find ASEG in fMRIPrep output first
    try:
        aseg_in_target_space = flexible_match(
            subject_id=subject_id,
            space=target_space,
            desc="aseg",
            suffix="dseg",
            session=session,
            run=run,
            logger=logger,
        )
    except FileNotFoundError:
        logger.info(f"No fMRIPrep ASEG found for {subject_id}, trying FreeSurfer ASEG...")
        try:
            # Try to convert FreeSurfer ASEG to T1w space
            aseg_in_target_space = convert_fs_aseg_to_t1w(
                subjects_dir=Path("/home/Archive/TOM/astronaut_data/data_management/preprocess_data/Output"),
                subject_id=subject_id,
                output_dir=output_dir,
                session=session,
                run=run,
                verbose=verbose,
            )
            if aseg_in_target_space is None:
                logger.error(f"Failed to convert FreeSurfer ASEG for {subject_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to convert FreeSurfer ASEG: {str(e)}")
            return None

    # Create output directory if needed
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate binary mask for the structure
    structure_mask = output_dir / f"{subject_id}_space-{target_space}_desc-{structure}_mask.nii.gz"
    try:
        _extract_structure_mask_t1(
            aseg_file=aseg_in_target_space,
            structure=structure,
            output_file=structure_mask,
            verbose=verbose,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"ASEG prep {type(e).__name__}: {str(e)}")
        return None

    # Generate surface from mask
    surface_file = output_dir / f"{subject_id}_space-{target_space}_desc-{structure}_surface.gii"
    try:
        _generate_surface_from_mask(
            mask_file=structure_mask,
            output_file=surface_file,
            verbose=verbose,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Surface generation {type(e).__name__}: {str(e)}")
        return None

    # Clean up intermediate files
    if structure_mask.exists():
        structure_mask.unlink()

    return surface_file

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
        fs_subject_dir = Path(subjects_dir) / f"sub-{subject_id_clean}" / "freesurfer"
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
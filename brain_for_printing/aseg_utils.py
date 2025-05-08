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

from .io_utils import run_cmd, flexible_match
from .mesh_utils import volume_to_gifti

L = logging.getLogger(__name__)

def _extract_structure_mask_t1(aseg_nifti_path: str, label_ids: List[int], output_mask_nifti_path: str, verbose: bool = False) -> bool:
    """
    Extract a binary mask from an ASEG NIfTI file for specific label IDs.
    
    Args:
        aseg_nifti_path: Path to input ASEG NIfTI file
        label_ids: List of label IDs to extract
        output_mask_nifti_path: Path to output binary mask
        verbose: Enable verbose logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not Path(aseg_nifti_path).exists():
        L.error(f"Input NIfTI not found: {aseg_nifti_path}")
        return False
        
    match_str = [str(lbl) for lbl in label_ids]
    L.info(f"Extracting {match_str} from {Path(aseg_nifti_path).name} -> {Path(output_mask_nifti_path).name}")
    
    try:
        run_cmd(["mri_binarize", "--i", aseg_nifti_path, "--match", *match_str, "--o", output_mask_nifti_path], verbose=verbose)
    except Exception as e:
        L.error(f"mri_binarize failed: {e}", exc_info=verbose)
        return False
        
    output_path = Path(output_mask_nifti_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        L.error(f"Output mask empty/not created: {output_path.name}")
        return False
        
    return True

def extract_structure_surface(
    subjects_dir: str,
    subject_id: str,
    label_ids: List[int],
    output_tag: str,
    space: str = 'T1',
    tmp_dir: str = '.',
    verbose: bool = False,
    session: Optional[str] = None,
    run: Optional[str] = None
) -> Optional[str]:
    """
    Extract a surface from an ASEG structure in specified space.
    
    Args:
        subjects_dir: Path to subjects directory
        subject_id: Subject ID (e.g., 'sub-01')
        label_ids: List of label IDs to extract
        output_tag: Tag for output files
        space: Target space ('T1' or 'MNI')
        tmp_dir: Temporary directory for intermediate files
        verbose: Enable verbose logging
        session: BIDS session ID
        run: BIDS run ID
        
    Returns:
        Optional[str]: Path to output GIFTI surface if successful, None otherwise
    """
    tmp_dir_path = Path(tmp_dir)
    tmp_dir_path.mkdir(parents=True, exist_ok=True)
    
    subject_id_clean = subject_id.replace('sub-', '')
    anat_dir = Path(subjects_dir) / f"sub-{subject_id_clean}" / "anat"
    
    output_mask_nii_path = tmp_dir_path / f"{output_tag}_mask_space-{space}_id-{uuid.uuid4().hex[:4]}.nii.gz"
    output_gii_path = tmp_dir_path / f"{output_tag}_space-{space}.surf.gii"
    
    aseg_in_target_space: Optional[str] = None
    
    try:
        if space.upper() == "T1":
            L.info(f"Locating T1-space aseg for {subject_id} ({output_tag})")
            aseg_in_target_space = flexible_match(
                base_dir=anat_dir,
                subject_id=subject_id,
                descriptor="desc-aseg",
                suffix="dseg",
                ext=".nii.gz",
                session=session,
                run=run,
                logger=L
            )
            L.info(f"Found T1-space aseg: {Path(aseg_in_target_space).name}")
            
        elif space.upper() == "MNI":
            L.info(f"Preparing MNI-space aseg for {subject_id} ({output_tag})")
            aseg_t1_path = flexible_match(
                base_dir=anat_dir,
                subject_id=subject_id,
                descriptor="desc-aseg",
                suffix="dseg",
                ext=".nii.gz",
                session=session,
                run=run,
                logger=L
            )
            L.debug(f"Found T1 aseg: {Path(aseg_t1_path).name}")
            
            xfm_t1_to_mni = flexible_match(
                base_dir=anat_dir,
                subject_id=subject_id,
                descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image",
                suffix="xfm",
                ext=".h5",
                session=session,
                run=run,
                logger=L
            )
            L.debug(f"Found T1->MNI xfm: {Path(xfm_t1_to_mni).name}")
            
            try:
                # Find MNI space reference geometry (use dseg if possible)
                mni_ref_path_str = flexible_match(
                    base_dir=anat_dir,
                    subject_id=subject_id,
                    space="MNI152NLin2009cAsym",
                    res="*",
                    descriptor="desc-aseg",
                    suffix="dseg",
                    ext=".nii.gz",
                    session=session,
                    run=run,
                    logger=L
                )
            except FileNotFoundError:
                try:
                    mni_ref_path_str = flexible_match(
                        base_dir=anat_dir,
                        subject_id=subject_id,
                        space="MNI152NLin2009cAsym",
                        res="*",
                        descriptor="preproc",
                        suffix="T1w",
                        ext=".nii.gz",
                        session=session,
                        run=run,
                        logger=L
                    )
                except FileNotFoundError:
                    L.warning(f"MNI ref not found for {subject_id}, using T1 aseg.")
                    mni_ref_path_str = aseg_t1_path
                    
            warped_aseg_path = tmp_dir_path / f"{output_tag}_aseg_in_mni_id-{uuid.uuid4().hex[:4]}.nii.gz"
            L.info(f"Warping {Path(aseg_t1_path).name} -> MNI ({warped_aseg_path.name})")
            
            run_cmd([
                "antsApplyTransforms",
                "-d", "3",
                "-i", aseg_t1_path,
                "-o", str(warped_aseg_path),
                "-r", mni_ref_path_str,
                "-t", xfm_t1_to_mni,
                "-n", "NearestNeighbor"
            ], verbose=verbose)
            
            aseg_in_target_space = str(warped_aseg_path)
            
        else:
            L.error(f"Unsupported ASEG space: {space}")
            return None
            
    except FileNotFoundError as e:
        L.error(f"ASEG prep FileNotFoundError: {e}", exc_info=verbose)
        return None
    except Exception as e:
        L.error(f"ASEG prep error: {e}", exc_info=verbose)
        return None
        
    if not aseg_in_target_space or not Path(aseg_in_target_space).exists():
        L.error(f"Aseg target space verified fail: '{aseg_in_target_space}'")
        return None
        
    success = _extract_structure_mask_t1(aseg_in_target_space, label_ids, str(output_mask_nii_path), verbose)
    if not success:
        L.error(f"Mask creation failed for {output_tag} in {space}.")
        return None
        
    try:
        volume_to_gifti(str(output_mask_nii_path), str(output_gii_path), level=0.5)
        L.info(f"Created GIFTI: {output_gii_path.name}")
        return str(output_gii_path)
    except Exception as e:
        L.error(f"NIfTI->GIFTI failed for {output_tag}: {e}", exc_info=verbose)
        return None 
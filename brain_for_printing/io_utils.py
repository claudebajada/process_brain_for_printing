# brain_for_printing/io_utils.py

"""
I/O and shell-utility helpers used throughout the package.
"""

from __future__ import annotations
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Union, List

import logging

L = logging.getLogger(__name__)

# Global flag for graceful exit
_should_exit = False

def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _should_exit
    if _should_exit:
        L.warning("Force exit requested. Terminating...")
        sys.exit(1)
    L.info("\nGracefully shutting down... (Press Ctrl+C again to force exit)")
    _should_exit = True

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def check_should_exit() -> bool:
    """Check if a graceful exit was requested."""
    return _should_exit

def validate_subject_data(subjects_dir: Union[str, Path], subject_id: str, required_files: Optional[List[str]] = None) -> bool:
    """
    Validate that required files exist for a subject.
    
    Args:
        subjects_dir: Path to subjects directory
        subject_id: Subject ID
        required_files: List of required file patterns to check
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    subjects_dir = Path(subjects_dir)
    subject_id = subject_id.replace('sub-', '')
    
    if not required_files:
        # Default required files
        required_files = [
            "anat/*T1w.nii.gz",  # T1w image
            f"sourcedata/freesurfer/sub-{subject_id}/mri/aseg.mgz"  # FreeSurfer ASEG
        ]
    
    missing_files = []
    for pattern in required_files:
        # Try both relative to subjects_dir and absolute path
        matches = list(subjects_dir.glob(pattern))
        if not matches:
            missing_files.append(pattern)
            
    if missing_files:
        L.error(f"Missing required files for subject {subject_id}:")
        for file in missing_files:
            L.error(f"  - {file}")
        return False
        
    return True

# --- External-process helpers ---
def run_cmd(cmd: List[str], verbose: bool = False) -> None:
    try:
        if check_should_exit():
            raise KeyboardInterrupt("Graceful exit requested")
            
        process = subprocess.run(
            cmd,
            check=True,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr if hasattr(exc, 'stderr') and exc.stderr else "<no stderr captured>"
        out_msg = exc.stdout if hasattr(exc, 'stdout') and exc.stdout else "<no stdout captured>"
        combined_msg = f"Stderr:\n{err_msg}\nStdout:\n{out_msg}"
        L.error(f"Cmd failed (Code {exc.returncode}): {' '.join(cmd)}\n{combined_msg}")
        raise RuntimeError(f"Cmd failed: {' '.join(cmd)}. See logs.") from exc
    except FileNotFoundError:
        L.error(f"Cmd not found: '{cmd[0]}'.")
        raise
    except KeyboardInterrupt:
        L.info("Command interrupted by user")
        raise
    except Exception as e:
        L.error(f"Unexpected error running: {' '.join(cmd)}\n{e}", exc_info=verbose)
        raise

def require_cmd(cmd: str, url_hint: str | None = None, logger=L) -> Path:
    path = shutil.which(cmd)
    if path: logger.debug(f"Cmd '{cmd}' found: {path}"); return Path(path)
    msg = f"Required command not found: '{cmd}'.";
    if url_hint: msg += f" See {url_hint}";
    logger.critical(msg); sys.exit(1)

def require_cmds(cmds: Iterable[str], url_hint: str | None = None, logger=L) -> dict[str, Path]:
    return {c: require_cmd(c, url_hint, logger) for c in cmds}

# --- Temporary-directory helper ---
@contextmanager
def temp_dir(tag: str = "bfp", keep: bool = False, base_dir: Optional[str] = None):
    prefix = f"_{tag}_{uuid.uuid4().hex[:6]}_"; td = None
    try:
        if keep: path = Path(tempfile.mkdtemp(prefix=prefix, dir=base_dir)); L.info(f"Created tmp dir (keep): {path}"); yield path
        else: td = tempfile.TemporaryDirectory(prefix=prefix, dir=base_dir); path = Path(td.name); L.info(f"Created tmp dir (delete): {path}"); yield path
    finally:
        if td:
            try: L.debug(f"Auto-cleaning tmp dir: {td.name}")
            except Exception as e_clean: L.error(f"Error auto-cleaning tmp dir {td.name}: {e_clean}")

# --- File-matching helpers ---
def first_match(pattern: str, logger=L) -> str:
    logger.debug(f"Searching for pattern: {pattern}")
    matches = glob.glob(pattern)
    if not matches: 
        logger.info(f"No files found matching: {pattern}")
        raise FileNotFoundError(f"No files found matching: {pattern}")
    # Sort matches for consistent selection when multiple exist
    matches.sort()
    if len(matches) > 1: 
        logger.warning(f"Multiple files ({len(matches)}) match '{pattern}'. Using first sorted: {matches[0]}")
    else: 
        logger.debug(f"Found unique match: {matches[0]}")
    return matches[0]

def flexible_match(
    base_dir: Union[str, os.PathLike],
    subject_id: str,
    session: Optional[str] = None,
    run: Optional[str] = None,
    space: Optional[str] = None,
    res: Optional[str] = None,
    hemi: Optional[str] = None,
    descriptor: Optional[str] = None,
    suffix: Optional[str] = None,
    ext: str = ".nii.gz",
    logger=L
) -> str:
    if not suffix: logger.error("flexible_match requires 'suffix'"); raise ValueError("Suffix required")
    base_dir_path = Path(base_dir)
    pattern = f"{subject_id if subject_id.startswith('sub-') else f'sub-{subject_id}'}"
    # Append entities with * separator ONLY if they are provided
    if session: pattern += f"*{session if session.startswith('ses-') else f'ses-{session}'}"
    if run: pattern += f"*{run if run.startswith('run-') else f'run-{run}'}"
    if space: pattern += f"*{space if space.startswith('space-') else f'space-{space}'}"
    if res: pattern += f"*{res if res == '*' or res.startswith('res-') else f'res-{res}'}"
    if hemi: pattern += f"*{hemi if hemi.startswith('hemi-') else f'hemi-{hemi}'}"
    # Append descriptor WITHOUT 'desc-' prefix if provided
    if descriptor: pattern += f"*{descriptor}"
    # Append suffix and extension, preceded by wildcard
    pattern += f"*{suffix}{ext}"
    full_pattern = str(base_dir_path / pattern)
    return first_match(full_pattern, logger=logger)

# --- nibabel convenience ---
def load_nifti(nifti_path):
    import nibabel as nib
    nii = nib.load(nifti_path); return nii.get_fdata(), nii.affine

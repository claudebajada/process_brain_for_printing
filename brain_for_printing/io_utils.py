# brain_for_printing/io_utils.py

import glob
import os
import subprocess
import re

def run_cmd(cmd, verbose=False):
    """
    Run an external command as a subprocess.
    If 'verbose' is True, the command and its output are shown.
    Otherwise, all external messages are suppressed.
    """
    if verbose:
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def first_match(pattern):
    """
    Return the first file that matches the given pattern.
    If multiple matches are found, return the first but warn the user.
    """
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[WARNING] Multiple files found for {pattern}; using {matches[0]}")
    return matches[0]

def flexible_match(base_dir, subject_id, descriptor, suffix, session=None, run=None, hemi=None, ext=".nii.gz"):
    """
    Flexibly match files following BIDS conventions.

    Parameters:
    - base_dir: base directory for files
    - subject_id: subject identifier (e.g., sub-01)
    - descriptor: BIDS descriptor (e.g., desc-aseg)
    - suffix: file suffix (e.g., dseg)
    - session: optional session identifier (e.g., ses-01)
    - run: optional run identifier (e.g., run-01)
    - hemi: optional hemisphere identifier (e.g., hemi-L)
    - ext: file extension (default ".nii.gz")

    Returns:
    - First matched file path
    """
    pattern = f"{base_dir}/{subject_id}"
    if session:
        pattern += f"_{session}"
    pattern += "*"
    if run:
        pattern += f"_{run}"
    if hemi:
        pattern += f"_{hemi}"
    if descriptor:
        pattern += f"_{descriptor}"
    if suffix:
        pattern += f"_{suffix}"
    pattern += ext

    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[WARNING] Multiple files found for {pattern}; using {matches[0]}")
    return matches[0]


def load_nifti(nifti_path):
    """
    Load NIfTI file and return data and affine.
    """
    import nibabel as nib
    nii = nib.load(nifti_path)
    return nii.get_fdata(), nii.affine


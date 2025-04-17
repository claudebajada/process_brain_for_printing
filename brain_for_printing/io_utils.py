# brain_for_printing/io_utils.py

"""
I/O and shell‑utility helpers used throughout the package.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

# --------------------------------------------------------------------- #
# External‑process helpers
# --------------------------------------------------------------------- #

def run_cmd(cmd: list[str] | tuple[str, ...], verbose: bool = False) -> None:
    """
    Execute *cmd* (a sequence) and raise with stderr attached if it fails.

    If *verbose* is ``True`` stdout/stderr are shown live; otherwise only
    captured stderr is printed on failure.
    """
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err = exc.stderr or "<no stderr captured>"
        raise RuntimeError(f"[CMD‑FAIL] {' '.join(cmd)}\n{err}") from None


def require_cmd(cmd: str, url_hint: str | None = None, logger=None) -> Path:
    """
    Abort the program if *cmd* is not on the user’s ``$PATH``.
    """
    path = shutil.which(cmd)
    if path:
        return Path(path)

    msg = f"Required external command not found: '{cmd}'."
    if url_hint:
        msg += f"  ({url_hint})"
    if logger:
        logger.error(msg)
    else:
        print(msg, file=sys.stderr)
    sys.exit(1)


def require_cmds(cmds: Iterable[str], url_hint: str | None = None, logger=None) -> dict[str, Path]:
    """Check several commands at once."""
    return {c: require_cmd(c, url_hint, logger) for c in cmds}


# --------------------------------------------------------------------- #
# Temporary‑directory helper
# --------------------------------------------------------------------- #

@contextmanager
def temp_dir(tag: str = "bfp", keep: bool = False, base_dir: str | None = None):
    """
    Context‑manager that yields a *Path* to a temporary directory
    (wrapper around ``tempfile.TemporaryDirectory``).

    If *keep* is True the folder is **not** deleted on exit (mirrors
    ``--no_clean`` CLI flags).
    """
    prefix = f"_{tag}_{uuid.uuid4().hex[:6]}_"
    if keep:
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=base_dir))
        try:
            yield path
        finally:
            pass                     # caller keeps it
    else:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=base_dir) as td:
            yield Path(td)
            # auto‑removed by TemporaryDirectory


# --------------------------------------------------------------------- #
# File‑matching helpers
# --------------------------------------------------------------------- #

def first_match(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    if len(matches) > 1:
        print(f"[WARNING] Multiple files for {pattern}; using {matches[0]}")
    return matches[0]


def flexible_match(
    base_dir: str | os.PathLike,
    subject_id: str,
    descriptor: str | None,
    suffix: str | None,
    session: str | None = None,
    run: str | None = None,
    hemi: str | None = None,
    ext: str = ".nii.gz",
) -> str:
    """
    Build a BIDS‑style glob pattern and return the first match.
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

    return first_match(pattern)


# --------------------------------------------------------------------- #
# nibabel convenience
# --------------------------------------------------------------------- #

def load_nifti(nifti_path):
    import nibabel as nib
    nii = nib.load(nifti_path)
    return nii.get_fdata(), nii.affine

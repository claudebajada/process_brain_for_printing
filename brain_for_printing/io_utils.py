# brain_for_printing/io_utils.py

import glob
import os
import subprocess

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


# brain_for_printing/log_utils.py

"""
Logging helpers + JSON run‑log writer.
"""

from __future__ import annotations
import datetime
import json
import logging # Make sure logging is imported
import os
import platform
import subprocess
from pathlib import Path

# MODIFIED: Add logger instance
L = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Real‑time logger
# --------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

def get_logger(
    name: str = "brain_for_printing",
    level: int = logging.INFO,
    log_path: str | None = None,
) -> logging.Logger:
    """
    Configure (or fetch) a module‑level logger.

    If *log_path* is given, messages go to that file; otherwise to stderr.
    Re‑using the same *name* returns the same configured instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger                      # already initialised

    logger.setLevel(level)
    handler = logging.FileHandler(log_path) if log_path else logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------- #
# JSON audit‑log (kept separate from the real‑time logger)
# --------------------------------------------------------------------- #

def write_log(log_dict: dict, output_dir: str | os.PathLike, base_name="run_log") -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dict["timestamp"] = timestamp
    log_dict["system_info"] = _get_system_info()
    log_dict["git_commit"] = _get_git_commit_hash()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_path = output_dir / f"{base_name}_{timestamp}.json"
    try: # Added try/except for file writing
        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)
        # MODIFIED: Changed print to L.info
        L.info(f"JSON log written => {log_path}")
    except Exception as e:
        L.error(f"Failed to write JSON log to {log_path}: {e}")


def _get_system_info() -> dict:
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _get_git_commit_hash() -> str | None:
    try:
        # Try getting hash from current directory (if it's the repo root or subdir)
        script_dir = Path(__file__).parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False, # Don't raise error if not a git repo or no HEAD
            cwd=script_dir # Check relative to script location first
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # If failed, try checking from the CWD (might be different from script dir)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False, # Don't raise error
            cwd=Path.cwd() # Check relative to current working directory
        )
        if result.returncode == 0:
            return result.stdout.strip()

    except FileNotFoundError: # Handle case where git command is not found
         L.debug("Git command not found, cannot get commit hash.")
         return None
    except Exception as e:
        L.debug(f"Could not get git commit hash: {e}")
        return None
    
    L.debug("Could not determine git commit hash.")
    return None # Explicitly return None if hash couldn't be found

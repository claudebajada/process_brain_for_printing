# brain_for_printing/log_utils.py

"""
Logging helpers + JSON run‑log writer.
"""

from __future__ import annotations
import datetime
import json
import logging
import os
import platform
import subprocess
from pathlib import Path

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
    with open(log_path, "w") as f:
        json.dump(log_dict, f, indent=2)

    print(f"[INFO] JSON log written ⇒ {log_path}")


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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None

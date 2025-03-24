# brain_for_printing/log_utils.py

import json
import os
import datetime
import platform
import subprocess

def write_log(log_dict, output_dir, base_name="run_log"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{base_name}_{timestamp}.json")
    log_dict["timestamp"] = timestamp
    log_dict["system_info"] = get_system_info()
    log_dict["git_commit"] = get_git_commit_hash()

    with open(log_path, "w") as f:
        json.dump(log_dict, f, indent=2)

    print(f"[INFO] Log written => {log_path}")


def get_system_info():
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }


def get_git_commit_hash():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


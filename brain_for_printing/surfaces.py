# brain_for_printing/surfaces.py

"""
surfaces.py
-----------
This module serves as a compatibility layer, re-exporting functions from specialized modules:
- surfgen_utils.py: Core surface generation functionality
- aseg_utils.py: ASEG-specific functions
- five_tt_utils.py: 5ttgen-related functions
"""
import logging
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .surfgen_utils import generate_brain_surfaces
from .aseg_utils import extract_structure_surface
from .five_tt_utils import run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes

L = logging.getLogger(__name__)

# Re-export functions from other modules for backward compatibility
__all__ = [
    'generate_brain_surfaces',
    'extract_structure_surface',
    'run_5ttgen_hsvs_save_temp_bids',
    'load_subcortical_and_ventricle_meshes'
]

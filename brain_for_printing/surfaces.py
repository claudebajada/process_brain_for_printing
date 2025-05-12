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

def export_surfaces(
    surfaces: Dict[str, trimesh.Trimesh],
    output_dir: Path,
    subject_id: str,
    space: str,
    preset: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Export surfaces to STL/OBJ files.
    
    Args:
        surfaces: Dictionary of surface meshes
        output_dir: Directory to save files
        subject_id: Subject ID
        space: Space (T1, MNI, etc.)
        preset: Optional preset name
        verbose: Enable verbose logging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, mesh in surfaces.items():
        if mesh is None:
            continue
            
        # Create filename
        space_suffix = f"_space-{space}" if space != "T1" else ""
        preset_suffix = f"_preset-{preset}" if preset else ""
        filename = f"{subject_id}{space_suffix}{preset_suffix}_{name}"
        
        # Export as STL
        stl_path = output_dir / f"{filename}.stl"
        mesh.export(stl_path)
        if verbose:
            L.info(f"Exported {stl_path}")
            
        # Export as OBJ
        obj_path = output_dir / f"{filename}.obj"
        mesh.export(obj_path)
        if verbose:
            L.info(f"Exported {obj_path}")

# Re-export functions from other modules for backward compatibility
__all__ = [
    'generate_brain_surfaces',
    'extract_structure_surface',
    'run_5ttgen_hsvs_save_temp_bids',
    'load_subcortical_and_ventricle_meshes',
    'export_surfaces'
]

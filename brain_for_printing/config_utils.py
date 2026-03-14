# brain_for_printing/config_utils.py

"""
Utilities for handling CLI configurations and presets.
"""

from typing import Dict, List, Set, Tuple
import logging

# Assuming constants are defined appropriately in brain_for_printing.constants
from . import constants as const

L = logging.getLogger(__name__)

# --- Presets Definition ---
# Defines standard sets of surfaces for generation. Keys are preset names,
# values are lists of surface keys/names recognized by the parsing logic below.
PRESETS: Dict[str, List[str]] = {
    "pial_brain": ['lh-pial', 'rh-pial', 'corpus_callosum', 'cerebellum', 'brainstem'],
    "white_brain": ['lh-white', 'rh-white', 'corpus_callosum', 'cerebellum_wm', 'brainstem'],
    "mid_brain": ['lh-mid', 'rh-mid', 'corpus_callosum', 'cerebellum', 'brainstem'],
    "cortical_pial": ['lh-pial', 'corpus_callosum', 'rh-pial'],
    "cortical_white": ['lh-white', 'corpus_callosum', 'rh-white'],
    "cortical_mid": ['lh-mid', 'corpus_callosum', 'rh-mid'],
    "brain_mask_surface": ["brain_mask_indicator"], # New preset
    # Add other presets as needed
}

def parse_preset(preset_name: str) -> Tuple[Set[str], Set[str], List[str]]:
    """
    Parses a preset name into required cortical types, other structure keys,
    and the exact list of mesh keys expected to be generated/loaded.

    Args:
        preset_name (str): The name of the preset (must be a key in PRESETS).

    Returns:
        Tuple[Set[str], Set[str], List[str]]:
            - Set of base cortical types needed (e.g., {"pial"}).
            - Set of other ASEG/CBM/BS/CC structure keys needed (e.g., {"brainstem"}).
            - List of all exact mesh keys expected (e.g., ["pial_L", "pial_R", "brainstem"]).

    Raises:
        ValueError: If the preset_name is not found in PRESETS.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found.")

    preset_list = PRESETS[preset_name]
    base_cortical_needed: Set[str] = set()
    other_structures_needed: Set[str] = set()
    exact_mesh_keys: List[str] = []

    # Handle special indicator for brain_mask_surface preset
    if preset_name == "brain_mask_surface" and preset_list == ["brain_mask_indicator"]:
        # This preset is handled differently in the CLI, no standard parsing needed here.
        # exact_mesh_keys can be set to indicate what the CLI should expect, e.g. ["brain_mask"]
        exact_mesh_keys.append("brain_mask")
        return base_cortical_needed, other_structures_needed, exact_mesh_keys

    for item in preset_list:
        # Check if it's an ASEG/CBM/BS/CC structure
        if item in const.CBM_BS_CC_CHOICES:
            other_structures_needed.add(item)
            exact_mesh_keys.append(item)
        # Check if it's a hemi-specific cortical surface
        elif item.startswith("lh-") or item.startswith("rh-"):
            try:
                hemi_prefix, base_type = item.split('-', 1)
                # Validate against defined cortical types in constants
                if base_type in const.CORTICAL_TYPES:
                    base_cortical_needed.add(base_type)
                    suffix = "_L" if hemi_prefix == 'lh' else "_R"
                    exact_mesh_keys.append(f"{base_type}{suffix}")
                else:
                    L.warning(f"Preset item '{item}' has invalid cortical type '{base_type}', skipping.")
            except ValueError:
                 L.warning(f"Preset item '{item}' is malformed, skipping.")
        # Check if it's a non-hemi cortical surface (implies both hemis)
        elif item in const.CORTICAL_TYPES:
             base_cortical_needed.add(item)
             exact_mesh_keys.append(f"{item}_L")
             exact_mesh_keys.append(f"{item}_R")
             L.debug(f"Preset item '{item}' implies both hemispheres.")
        else:
            L.warning(f"Preset item '{item}' not recognized as cortical or CBM/BS/CC, skipping.")

    # Return sorted lists/sets for consistency
    return base_cortical_needed, other_structures_needed, sorted(exact_mesh_keys)

# --- Add other shared config/CLI utilities here if needed ---

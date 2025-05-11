#!/usr/bin/env python
# brain_for_printing/cli_cortical_surfaces.py
#
# CLI for generating brain surfaces from different categories.
# Supports T1, MNI, or target subject space.
# Uses MRtrix-style warping for non-T1 spaces.

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import trimesh
import glob
import os
import numpy as np
from typing import Dict, Optional, List, Set, Tuple, Union

# --- Local Imports ---
from .io_utils import temp_dir, require_cmds, flexible_match
from .log_utils import get_logger, write_log
from .surfaces import generate_brain_surfaces, run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes
from . import constants as const
from .config_utils import PRESETS, parse_preset
# --- End Imports ---

L = logging.getLogger("brain_for_printing_surfaces")

# Helper function from previous version (parsing custom surface args)
def parse_custom_surface_args(cortical_request: List[str], cbm_bs_cc_request: List[str]) -> Tuple[Set[str], Set[str], List[str]]:
    """
    Parses custom --cortical-surfaces and --cbm-bs-cc args.

    Returns:
        Tuple[Set[str], Set[str], List[str]]:
            - Set of base cortical types needed (e.g., {"pial"}).
            - Set of other CBM/BS/CC structure keys needed (e.g., {"brainstem"}).
            - List of exact mesh keys derived from requests (e.g., ["pial_L", "pial_R", "brainstem"]).
    """
    base_cortical_needed: Set[str] = set()
    cbm_bs_cc_needed: Set[str] = set()
    exact_mesh_keys: List[str] = []
    for req_surf in cortical_request:
        if req_surf in const.CORTICAL_TYPES:
             base_cortical_needed.add(req_surf)
             exact_mesh_keys.append(f"{req_surf}_L")
             exact_mesh_keys.append(f"{req_surf}_R")
        elif req_surf in const.HEMI_CORTICAL_TYPES:
            try:
                hemi_prefix, base_type = req_surf.split('-', 1)
                if hemi_prefix in ['lh', 'rh'] and base_type in const.CORTICAL_TYPES:
                    base_cortical_needed.add(base_type)
                    suffix = "_L" if hemi_prefix == 'lh' else "_R"
                    exact_mesh_keys.append(f"{base_type}{suffix}")
                else:
                    L.warning(f"Ignoring malformed hemi surf: {req_surf}")
            except ValueError:
                L.warning(f"Ignoring malformed hemi surf: {req_surf}")
        else:
            L.warning(f"Ignoring unrecognized cortical surf: {req_surf}")
    for req_other in cbm_bs_cc_request:
        if req_other in const.CBM_BS_CC_CHOICES:
            cbm_bs_cc_needed.add(req_other)
            exact_mesh_keys.append(req_other)
        else:
            L.warning(f"Ignoring unrecognized cbm-bs-cc surf: {req_other}")
    return base_cortical_needed, cbm_bs_cc_needed, sorted(list(set(exact_mesh_keys)))


# --------------------------------------------------------------------------- #
# CLI Argument Parser Setup
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate brain surfaces.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory (e.g., /path/to/bids/derivatives).")
    parent_parser.add_argument("--work_dir", default=None, help="Optional base directory for intermediate work files. Defaults to a 'work' directory alongside 'derivatives'.")
    parent_parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-CB).")
    parent_parser.add_argument("--output_dir", default=".", help="Output directory for final STL/OBJ files.")
    parent_parser.add_argument("--space", default="T1", help="Output space: 'T1' (native), 'MNI' (template), or target subject ID (e.g., 'sub-XYZ') to warp into that subject's T1 space.")
    parent_parser.add_argument("--run", default=None, help="BIDS run entity.")
    parent_parser.add_argument("--session", default=None, help="BIDS session entity.")
    parent_parser.add_argument("--split_outputs", action="store_true", help="Export surfaces separately.")
    parent_parser.add_argument("--no_clean", action="store_true", help="Keep temporary folder used for generation/warping.")
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")

    subparsers = parser.add_subparsers(dest="mode", required=True, title="Modes")

    parser_custom = subparsers.add_parser("custom", help="Specify surfaces manually.", parents=[parent_parser])
    parser_custom.add_argument("--cortical-surfaces", "--cortical_surfaces", nargs='*', default=[], choices=const.CORTICAL_CHOICES, metavar='SURFACE', help=f"Cortical surfaces. Choices: {const.CORTICAL_CHOICES}")
    sgm_help_text = (f"VTK Subcortical Gray. Use '{const.VTK_KEYWORDS[0]}' or specific names (e.g., 'L_Thal'). Common examples: {', '.join(const.SGM_NAME_EXAMPLES)}. Requires 5ttgen data.")
    parser_custom.add_argument("--subcortical-gray", "--subcortical_gray", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=sgm_help_text)
    parser_custom.add_argument("--cbm-bs-cc", nargs='*', default=[], choices=const.CBM_BS_CC_CHOICES, metavar='STRUCTURE', help=f"ASEG Cerebellum, BS, CC. Choices: {const.CBM_BS_CC_CHOICES}")
    vent_help_text = ( f"VTK Ventricles/Vessels. Use '{const.VTK_KEYWORDS[0]}' or specific names (e.g., '3rd-Ventricle'). Common examples: {', '.join(const.VENT_NAME_EXAMPLES)}. Requires 5ttgen data.")
    parser_custom.add_argument("--ventricular-system", "--ventricular_system", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=vent_help_text)
    parser_custom.add_argument("--no-fill-structures", "--no_fill_structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip fill for CBM/BS/CC.")
    parser_custom.add_argument("--no-smooth-structures", "--no_smooth_structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip smooth for CBM/BS/CC.")

    parser_preset = subparsers.add_parser("preset", help="Use presets (cortical & CBM/BS/CC).", parents=[parent_parser])
    parser_preset.add_argument("--name", required=True, choices=list(PRESETS.keys()), help="Preset name.")
    parser_preset.add_argument("--no-fill-structures", "--no_fill_structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Preset override: skip fill.")
    parser_preset.add_argument("--no-smooth-structures", "--no_smooth_structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Preset override: skip smooth.")

    return parser

# --------------------------------------------------------------------------- #
# Main Execution Logic
# --------------------------------------------------------------------------- #
def main():
    """Main entry point for the cortical surfaces CLI."""
    args = _build_parser().parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    L = logging.getLogger(__name__)
    
    try:
        # Validate subject data before proceeding
        if not validate_subject_data(args.subjects_dir, args.subject_id):
            L.error("Required files missing. Please check the paths and try again.")
            sys.exit(1)
            
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate surfaces
        surfaces = generate_brain_surfaces(
            subjects_dir=args.subjects_dir,
            subject_id=args.subject_id,
            space=args.space,
            surfaces=args.surfaces,
            extract_structures=args.extract_structures,
            no_fill_structures=args.no_fill_structures,
            no_smooth_structures=args.no_smooth_structures,
            run=args.run,
            session=args.session,
            verbose=args.verbose,
            tmp_dir=args.tmp_dir,
            preloaded_vtk_meshes=args.preloaded_vtk_meshes
        )
        
        if not surfaces:
            L.error("Failed to generate surfaces")
            sys.exit(1)
            
        # Export surfaces
        export_surfaces(
            surfaces=surfaces,
            output_dir=output_dir,
            subject_id=args.subject_id,
            space=args.space,
            preset=args.preset,
            verbose=args.verbose
        )
        
        L.info("Script finished.")
        
    except KeyboardInterrupt:
        L.info("\nScript interrupted by user")
        sys.exit(0)
    except Exception as e:
        L.error(f"An error occurred: {str(e)}", exc_info=args.verbose)
        sys.exit(1)

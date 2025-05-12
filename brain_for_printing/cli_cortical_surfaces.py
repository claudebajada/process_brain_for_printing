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
from .io_utils import temp_dir, require_cmds, flexible_match, validate_subject_data
# MODIFIED: Import write_log
from .log_utils import get_logger, write_log
from .surfaces import (
    generate_brain_surfaces,
    run_5ttgen_hsvs_save_temp_bids,
    load_subcortical_and_ventricle_meshes,
    export_surfaces
)
from . import constants as const
from .config_utils import PRESETS, parse_preset
# --- End Imports ---

# L = logging.getLogger("brain_for_printing_surfaces") # Keep logger setup in main

# Helper function from previous version (parsing custom surface args)
def parse_custom_surface_args(cortical_request: List[str], cbm_bs_cc_request: List[str], logger: logging.Logger) -> Tuple[Set[str], Set[str], List[str]]: # Added logger pass-through
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
                    logger.warning(f"Ignoring malformed hemi surf: {req_surf}") # Use passed logger
            except ValueError:
                logger.warning(f"Ignoring malformed hemi surf: {req_surf}") # Use passed logger
        else:
            logger.warning(f"Ignoring unrecognized cortical surf: {req_surf}") # Use passed logger
    for req_other in cbm_bs_cc_request:
        if req_other in const.CBM_BS_CC_CHOICES:
            cbm_bs_cc_needed.add(req_other)
            exact_mesh_keys.append(req_other)
        else:
            logger.warning(f"Ignoring unrecognized cbm-bs-cc surf: {req_other}") # Use passed logger
    return base_cortical_needed, cbm_bs_cc_needed, sorted(list(set(exact_mesh_keys)))


# --------------------------------------------------------------------------- #
# CLI Argument Parser Setup
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate brain surfaces.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory (e.g., /path/to/bids/derivatives).")
    # MODIFIED: Added work_dir help text clarification
    parent_parser.add_argument("--work_dir", default=None, help="Optional base directory for intermediate work files. If not set, a temporary directory is created and removed unless --no_clean is used.")
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
    # MODIFIED: Use get_logger for consistency
    log_level = logging.DEBUG if args.verbose else logging.INFO
    L = get_logger(__name__, level=log_level) # Use the helper

    # MODIFIED: Initialize runlog dictionary
    runlog = {
        "tool": f"brain_for_printing_cortical_surfaces_{args.mode}",
        "args": vars(args), # Store all arguments
        "steps": [],
        "warnings": [],
        "output_dir": os.path.abspath(args.output_dir),
        "output_files": []
    }

    # MODIFIED: Define work_dir behavior more clearly
    # If work_dir is provided, use it. Otherwise, use temp_dir context manager.
    custom_work_dir = Path(args.work_dir) if args.work_dir else None
    if custom_work_dir:
        custom_work_dir.mkdir(parents=True, exist_ok=True)
        L.info(f"Using specified work directory: {custom_work_dir}")
        runlog["steps"].append(f"Using specified work directory: {custom_work_dir}")
        temp_work_dir = str(custom_work_dir) # Use the specified path directly
    else:
        L.info("Using temporary directory context manager.")
        # The actual path will be determined by the `temp_dir` context manager below

    try:
        # Validate subject data before proceeding
        L.info(f"Validating data for {args.subject_id} in {args.subjects_dir}")
        if not validate_subject_data(args.subjects_dir, args.subject_id):
            # MODIFIED: Add to runlog before exiting
            runlog["warnings"].append("Required files missing.")
            L.error("Required files missing. Please check the paths and try again.")
            # MODIFIED: Write log on error exit
            if custom_work_dir: # If work_dir was specified, write log there
                 write_log(runlog, custom_work_dir, base_name="cortical_surfaces_failed_log")
            else: # Otherwise write to output_dir
                 write_log(runlog, args.output_dir, base_name="cortical_surfaces_failed_log")
            sys.exit(1)
        runlog["steps"].append("Subject data validation passed")

        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse surface requests based on mode
        L.info(f"Parsing surface requests for mode: {args.mode}")
        if args.mode == "custom":
            # MODIFIED: Pass logger to parse function
            base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys = parse_custom_surface_args(
                args.cortical_surfaces if hasattr(args, 'cortical_surfaces') else [],
                args.cbm_bs_cc if hasattr(args, 'cbm_bs_cc') else [],
                L # Pass the logger
            )
            runlog["preset_name"] = "custom"
        else:  # preset mode
            base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys = parse_preset(args.name)
            runlog["preset_name"] = args.name # Add preset name to log

        runlog["requested_cortical_types"] = sorted(list(base_cortical_needed))
        runlog["requested_cbm_bs_cc"] = sorted(list(cbm_bs_cc_needed))
        runlog["expected_mesh_keys"] = exact_mesh_keys
        L.info(f"Base cortical types needed: {base_cortical_needed}")
        L.info(f"CBM/BS/CC structures needed: {cbm_bs_cc_needed}")
        L.info(f"Exact mesh keys expected: {exact_mesh_keys}")
        runlog["steps"].append("Parsed surface requests")

        # MODIFIED: Handle temporary directory context OR specified work_dir
        if custom_work_dir:
            # --- Generate surfaces using the specified work_dir ---
            L.info(f"Generating surfaces in space '{args.space}' using work_dir: {temp_work_dir}")
            surfaces = generate_brain_surfaces(
                subjects_dir=args.subjects_dir,
                subject_id=args.subject_id,
                space=args.space,
                surfaces=tuple(base_cortical_needed),
                extract_structures=list(cbm_bs_cc_needed),
                no_fill_structures=args.no_fill_structures,
                no_smooth_structures=args.no_smooth_structures,
                run=args.run,
                session=args.session,
                verbose=args.verbose,
                tmp_dir=temp_work_dir, # Pass the specified work_dir
                preloaded_vtk_meshes={}
            )
            runlog["steps"].append(f"Surface generation process completed in {temp_work_dir}")
        else:
             # --- Generate surfaces using the temp_dir context manager ---
            with temp_dir("cortical_surf_gen", keep=args.no_clean, base_dir=args.output_dir) as temp_context_dir:
                 L.info(f"Generating surfaces in space '{args.space}' using temp_dir: {temp_context_dir}")
                 temp_work_dir = str(temp_context_dir) # Define temp_work_dir within context
                 runlog["steps"].append(f"Created temp dir: {temp_work_dir}")
                 surfaces = generate_brain_surfaces(
                     subjects_dir=args.subjects_dir,
                     subject_id=args.subject_id,
                     space=args.space,
                     surfaces=tuple(base_cortical_needed),
                     extract_structures=list(cbm_bs_cc_needed),
                     no_fill_structures=args.no_fill_structures,
                     no_smooth_structures=args.no_smooth_structures,
                     run=args.run,
                     session=args.session,
                     verbose=args.verbose,
                     tmp_dir=temp_work_dir, # Pass the temp context dir
                     preloaded_vtk_meshes={}
                 )
                 runlog["steps"].append(f"Surface generation process completed in {temp_work_dir}")

                 # MODIFIED: Add warning if temp folder is kept
                 if args.no_clean:
                    runlog["warnings"].append(f"Temporary folder retained: {temp_work_dir}")
                    L.warning(f"Temporary folder retained: {temp_work_dir}")


        if not surfaces or not any(surfaces.values()): # Check if dict is empty or all values are None
            # MODIFIED: Add warning to log
            runlog["warnings"].append("Surface generation yielded no valid meshes.")
            L.error("Failed to generate surfaces or all generated surfaces were empty.")
             # MODIFIED: Write log on error exit
            log_output_location = custom_work_dir if custom_work_dir else args.output_dir
            write_log(runlog, log_output_location, base_name="cortical_surfaces_failed_log")
            sys.exit(1)

        # Filter out None or empty meshes before exporting
        valid_surfaces = {k: v for k, v in surfaces.items() if v is not None and not v.is_empty}
        if not valid_surfaces:
             runlog["warnings"].append("All generated surfaces were empty after filtering.")
             L.error("All generated surfaces were empty after filtering.")
             log_output_location = custom_work_dir if custom_work_dir else args.output_dir
             write_log(runlog, log_output_location, base_name="cortical_surfaces_failed_log")
             sys.exit(1)
        runlog["generated_surface_keys"] = sorted(list(valid_surfaces.keys()))
        L.info(f"Successfully generated {len(valid_surfaces)} non-empty surfaces.")
        runlog["steps"].append(f"Successfully generated {len(valid_surfaces)} non-empty surfaces.")

        # --- Export surfaces ---
        L.info(f"Exporting {len(valid_surfaces)} surfaces to {output_dir} (split={args.split_outputs})")
        # Use the original surfaces dict for export in case logging needs info about failed ones later
        export_surfaces(
            surfaces=surfaces, # Pass the original dict
            output_dir=output_dir,
            subject_id=args.subject_id,
            space=args.space,
            preset=args.name if args.mode == "preset" else None,
            verbose=args.verbose,
            split_outputs=args.split_outputs,
            file_format="stl"  # cortical_surfaces always uses STL
        )
        # MODIFIED: Log output files (Need to get paths from export_surfaces or assume naming convention)
        # Assuming export_surfaces doesn't return paths, we can infer them
        # This part might need adjustment if export_surfaces changes
        if args.split_outputs:
            for name in valid_surfaces.keys():
                space_suffix = f"_space-{args.space}" if args.space != "T1" else ""
                preset_suffix = f"_preset-{runlog['preset_name']}" if runlog['preset_name'] != 'custom' else ""
                filename = f"{args.subject_id}{space_suffix}{preset_suffix}_{name}.stl"
                runlog["output_files"].append(str(output_dir / filename))
        else:
            space_suffix = f"_space-{args.space}" if args.space != "T1" else ""
            preset_suffix = f"_preset-{runlog['preset_name']}" if runlog['preset_name'] != 'custom' else ""
            filename = f"{args.subject_id}{space_suffix}{preset_suffix}_combined.stl"
            runlog["output_files"].append(str(output_dir / filename))
        runlog["steps"].append("Export process completed.")

        # MODIFIED: Write log on success
        # Write log to work_dir if specified and kept, otherwise output_dir
        log_output_location = custom_work_dir if custom_work_dir and args.no_clean else args.output_dir
        write_log(runlog, log_output_location, base_name="cortical_surfaces_log")
        L.info("Script finished successfully.")

    except KeyboardInterrupt:
        L.info("\nScript interrupted by user")
        # MODIFIED: Add log write on interrupt
        runlog["warnings"].append("Script interrupted by user (KeyboardInterrupt).")
        log_output_location = custom_work_dir if custom_work_dir else args.output_dir
        write_log(runlog, log_output_location, base_name="cortical_surfaces_interrupted_log")
        sys.exit(0)
    except Exception as e:
        L.error(f"An unexpected error occurred: {str(e)}", exc_info=args.verbose)
        # MODIFIED: Add log write on general exception
        runlog["warnings"].append(f"An unexpected error occurred: {str(e)}")
        log_output_location = custom_work_dir if custom_work_dir else args.output_dir
        write_log(runlog, log_output_location, base_name="cortical_surfaces_error_log")
        sys.exit(1)

if __name__ == "__main__":
    main()

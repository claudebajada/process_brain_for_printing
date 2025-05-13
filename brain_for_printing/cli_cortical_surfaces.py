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
import os # Keep os import for os.path.abspath
import numpy as np # Keep numpy if still needed elsewhere, or remove
from typing import Dict, Optional, List, Set, Tuple, Union
from contextlib import contextmanager # For the dummy context manager

# --- Local Imports ---
from .io_utils import temp_dir, require_cmds, flexible_match, validate_subject_data
from .log_utils import get_logger, write_log
from .surfaces import (
    generate_brain_surfaces,
    export_surfaces
)
# Import for brain mask surface generation
from .surfgen_utils import generate_single_brain_mask_surface # Corrected import path
from . import constants as const
from .config_utils import PRESETS, parse_preset
# --- End Imports ---

# Helper function from previous version (parsing custom surface args)
def parse_custom_surface_args(cortical_request: List[str], cbm_bs_cc_request: List[str], logger: logging.Logger) -> Tuple[Set[str], Set[str], List[str]]:
    base_cortical_needed: Set[str] = set()
    cbm_bs_cc_needed: Set[str] = set()
    exact_mesh_keys: List[str] = []
    for req_surf in cortical_request or []: # Ensure iterable
        if req_surf in const.CORTICAL_TYPES:
             base_cortical_needed.add(req_surf)
             exact_mesh_keys.extend([f"{req_surf}_L", f"{req_surf}_R"])
        elif req_surf in const.HEMI_CORTICAL_TYPES:
            try:
                hemi_prefix, base_type = req_surf.split('-', 1)
                if hemi_prefix in ['lh', 'rh'] and base_type in const.CORTICAL_TYPES:
                    base_cortical_needed.add(base_type)
                    suffix = "_L" if hemi_prefix == 'lh' else "_R"
                    exact_mesh_keys.append(f"{base_type}{suffix}")
                else: logger.warning(f"Ignoring malformed hemi surf: {req_surf}") 
            except ValueError: logger.warning(f"Ignoring malformed hemi surf: {req_surf}") 
        else: logger.warning(f"Ignoring unrecognized cortical surf: {req_surf}") 

    for req_other in cbm_bs_cc_request or []: # Ensure iterable
        if req_other in const.CBM_BS_CC_CHOICES:
            cbm_bs_cc_needed.add(req_other)
            exact_mesh_keys.append(req_other)
        else: logger.warning(f"Ignoring unrecognized cbm-bs-cc surf: {req_other}") 
    return base_cortical_needed, cbm_bs_cc_needed, sorted(list(set(exact_mesh_keys)))


# --------------------------------------------------------------------------- #
# CLI Argument Parser Setup
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate brain surfaces.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory.")
    parent_parser.add_argument("--work_dir", default=None, help="Optional base directory for intermediate work files.")
    parent_parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-CB).")
    parent_parser.add_argument("--output_dir", default=".", help="Output directory for final STL files.")
    parent_parser.add_argument("--space", default="T1", help="Output space: 'T1', 'MNI', or target subject ID.")
    parent_parser.add_argument("--run", default=None, help="BIDS run entity.")
    parent_parser.add_argument("--session", default=None, help="BIDS session entity.")
    parent_parser.add_argument("--split_outputs", action="store_true", help="Export surfaces separately.")
    parent_parser.add_argument("--no_clean", action="store_true", help="Keep temporary folder.")
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")

    subparsers = parser.add_subparsers(dest="mode", required=True, title="Modes")

    # --- Custom Mode Parser ---
    parser_custom = subparsers.add_parser("custom", help="Specify surfaces manually.", parents=[parent_parser])
    parser_custom.add_argument("--cortical-surfaces", "--cortical_surfaces", nargs='*', default=[], choices=const.CORTICAL_CHOICES, metavar='SURFACE', help=f"Cortical surfaces. Choices: {const.CORTICAL_CHOICES}")
    # Removed SGM and Ventricular for this iteration as they require 5ttgen logic not fully integrated here.
    # sgm_help_text = (f"VTK Subcortical Gray. Use '{const.VTK_KEYWORDS[0]}' or specific names. Requires 5ttgen data.")
    # parser_custom.add_argument("--subcortical-gray", "--subcortical_gray", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=sgm_help_text)
    parser_custom.add_argument("--cbm-bs-cc", nargs='*', default=[], choices=const.CBM_BS_CC_CHOICES, metavar='STRUCTURE', help=f"ASEG Cerebellum, BS, CC. Choices: {const.CBM_BS_CC_CHOICES}")
    # vent_help_text = ( f"VTK Ventricles/Vessels. Use '{const.VTK_KEYWORDS[0]}' or specific names. Requires 5ttgen data.")
    # parser_custom.add_argument("--ventricular-system", "--ventricular_system", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=vent_help_text)
    parser_custom.add_argument("--no-fill-structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip fill for specified CBM/BS/CC structures.")
    parser_custom.add_argument("--no-smooth-structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip smooth for specified CBM/BS/CC structures.")
    parser_custom.add_argument('--generate_brain_mask', action='store_true', help="Generate a surface from the brain mask (inflation off, smoothing on by default).")

    # --- Preset Mode Parser ---
    parser_preset = subparsers.add_parser("preset", help="Use presets.", parents=[parent_parser])
    parser_preset.add_argument("--name", required=True, choices=list(PRESETS.keys()), help="Preset name.")
    # Removed --no-fill-structures and --no-smooth-structures from preset mode
    # Removed --brain_mask_inflate_mm and --brain_mask_no_smooth

    return parser

# --------------------------------------------------------------------------- #
# Main Execution Logic
# --------------------------------------------------------------------------- #
def main():
    args = _build_parser().parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    L = get_logger("brain_for_printing_surfaces", level=log_level)

    runlog = {
        "tool": f"brain_for_printing_cortical_surfaces_{args.mode}",
        "args": {k: v for k, v in vars(args).items() if v is not None},
        "steps": [],
        "warnings": [],
        "output_dir": os.path.abspath(args.output_dir),
        "output_files": []
    }

    custom_work_dir_path = Path(args.work_dir) if args.work_dir else None
    if custom_work_dir_path:
        custom_work_dir_path.mkdir(parents=True, exist_ok=True)
        L.info(f"Using specified work directory: {custom_work_dir_path}")
        runlog["steps"].append(f"Using specified work directory: {custom_work_dir_path}")
    else:
        L.info("Using temporary directory context manager for work directory.")

    try:
        L.info(f"Validating data for {args.subject_id} in {args.subjects_dir}")
        subject_path_to_check = Path(args.subjects_dir) / args.subject_id
        if not Path(args.subjects_dir).is_dir() or not subject_path_to_check.is_dir():
             L.error(f"Subjects directory or subject ID subdirectory not found: {subject_path_to_check}")
             runlog["warnings"].append("Subject directory not found.")
             # Attempt to write log even on early exit
             log_output_location_on_fail = custom_work_dir_path if custom_work_dir_path else args.output_dir
             write_log(runlog, str(log_output_location_on_fail), base_name="cortical_surfaces_failed_log")
             sys.exit(1)
        runlog["steps"].append("Initial subject directory validation passed")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        @contextmanager
        def get_work_dir_context_manager(custom_dir, keep_temp, base_out_dir_for_temp):
            if custom_dir:
                yield str(custom_dir) # Yield the path string directly
            else:
                with temp_dir("cortical_surf_gen", keep=keep_temp, base_dir=str(base_out_dir_for_temp)) as td_path:
                    yield td_path

        with get_work_dir_context_manager(custom_work_dir_path, args.no_clean, output_dir) as work_dir_str:
            work_dir = Path(work_dir_str) # work_dir is now always a Path object
            L.info(f"Active work directory: {work_dir}")
            runlog["steps"].append(f"Active work directory: {work_dir}")

            surfaces_to_export: Dict[str, Optional[trimesh.Trimesh]] = {}
            preset_name_for_log = args.name if args.mode == "preset" else "custom"

            # Flag to track if brain mask was requested and handled
            brain_mask_processed = False

            # --- Brain Mask Surface Generation (Preset or Custom) ---
            if (args.mode == "preset" and args.name == "brain_mask_surface") or \
               (args.mode == "custom" and args.generate_brain_mask):
                
                if args.mode == "preset":
                    L.info("Processing 'brain_mask_surface' preset...")
                    runlog["steps"].append("Selected 'brain_mask_surface' preset")
                    preset_name_for_log = "brain_mask_surface"
                else: # custom mode with --generate_brain_mask
                    L.info("Processing custom request for brain_mask surface...")
                    runlog["steps"].append("Custom request: brain_mask_surface")
                    preset_name_for_log = "custom_brain_mask"


                brain_mask_mesh = generate_single_brain_mask_surface(
                    subjects_dir=args.subjects_dir,
                    subject_id=args.subject_id,
                    space=args.space,
                    inflate_mm=0.0,  # Hardcoded: no inflation
                    no_smooth=False, # Hardcoded: smoothing is ON
                    run=args.run,
                    session=args.session,
                    tmp_dir=work_dir,
                    logger=L,
                    verbose=args.verbose
                )
                if brain_mask_mesh and not brain_mask_mesh.is_empty:
                    surfaces_to_export["brain_mask"] = brain_mask_mesh
                    runlog["steps"].append("Generated brain mask surface")
                    L.info("Successfully generated brain mask surface.")
                else:
                    L.warning("Brain mask surface generation failed or resulted in an empty mesh.")
                    runlog["warnings"].append("Brain mask surface generation failed or empty.")
                brain_mask_processed = True


            # --- Standard Surface Generation (if not exclusively brain_mask_surface preset) ---
            if not (args.mode == "preset" and args.name == "brain_mask_surface"):
                # This block handles:
                # 1. All other presets.
                # 2. Custom mode requests *other than or in addition to* brain_mask.
                
                L.info(f"Parsing standard surface requests for mode: {args.mode}")
                if args.mode == "custom":
                    # For custom mode, if generate_brain_mask was true, it's already handled.
                    # We now process other custom requests like cortical, cbm-bs-cc.
                    cortical_req = args.cortical_surfaces
                    cbm_req = args.cbm_bs_cc
                    # Only parse and generate if there are actual requests beyond brain_mask
                    if cortical_req or cbm_req:
                        base_cortical_needed, cbm_bs_cc_needed, _ = parse_custom_surface_args(
                            cortical_req, cbm_req, L
                        )
                        preset_name_for_log = "custom" # Override if it was custom_brain_mask
                    else: # No other custom requests
                        base_cortical_needed, cbm_bs_cc_needed = set(), set()

                else:  # preset mode (but not brain_mask_surface, already handled)
                    base_cortical_needed, cbm_bs_cc_needed, _ = parse_preset(args.name)
                    preset_name_for_log = args.name
                
                runlog["requested_cortical_types"] = sorted(list(base_cortical_needed))
                runlog["requested_cbm_bs_cc"] = sorted(list(cbm_bs_cc_needed))

                if base_cortical_needed or cbm_bs_cc_needed: # Proceed if there's something to generate
                    L.info(f"Base cortical types needed: {base_cortical_needed}")
                    L.info(f"CBM/BS/CC structures needed: {cbm_bs_cc_needed}")
                    runlog["steps"].append("Parsed standard surface requests")

                    # Determine no_fill and no_smooth based on mode
                    # For presets, these are empty lists (i.e., fill and smooth by default)
                    # For custom, they come from args.
                    no_fill = args.no_fill_structures if args.mode == "custom" else []
                    no_smooth = args.no_smooth_structures if args.mode == "custom" else []

                    generated_standard_surfaces = generate_brain_surfaces(
                        subjects_dir=args.subjects_dir,
                        subject_id=args.subject_id,
                        space=args.space,
                        surfaces=tuple(base_cortical_needed),
                        extract_structures=list(cbm_bs_cc_needed),
                        no_fill_structures=no_fill,
                        no_smooth_structures=no_smooth,
                        run=args.run,
                        session=args.session,
                        verbose=args.verbose,
                        tmp_dir=str(work_dir),
                        preloaded_vtk_meshes={}
                    )
                    runlog["steps"].append(f"Standard surface generation process completed in {work_dir}")
                    
                    valid_standard_surfaces = {
                        k: v for k, v in generated_standard_surfaces.items() if v is not None and not v.is_empty
                    }
                    surfaces_to_export.update(valid_standard_surfaces)
                elif not brain_mask_processed: # No standard surfaces and brain mask wasn't processed
                     L.info("No surfaces requested for generation.")
                     runlog["steps"].append("No surfaces requested for generation.")


            # --- Exporting ---
            if not surfaces_to_export or not any(s is not None and not s.is_empty for s in surfaces_to_export.values()):
                runlog["warnings"].append("Surface generation yielded no valid meshes to export.")
                L.error("Failed to generate surfaces or all generated surfaces were empty.")
            else:
                num_valid_to_export = sum(1 for s in surfaces_to_export.values() if s is not None and not s.is_empty)
                runlog["generated_surface_keys"] = sorted([k for k, v in surfaces_to_export.items() if v is not None and not v.is_empty])
                L.info(f"Successfully generated/processed {num_valid_to_export} non-empty surfaces for export.")
                runlog["steps"].append(f"Successfully generated/processed {num_valid_to_export} non-empty surfaces for export.")

                L.info(f"Exporting {num_valid_to_export} surfaces to {output_dir} (split={args.split_outputs})")
                export_surfaces(
                    surfaces=surfaces_to_export, # Pass the dict with potentially None/empty meshes
                    output_dir=output_dir,
                    subject_id=args.subject_id,
                    space=args.space,
                    preset=preset_name_for_log,
                    verbose=args.verbose,
                    split_outputs=args.split_outputs,
                    file_format="stl"
                )
                
                # Infer output file names for the log
                if args.split_outputs:
                    for name, mesh_obj in surfaces_to_export.items():
                        if mesh_obj is not None and not mesh_obj.is_empty:
                            space_suffix = f"_space-{args.space}" if args.space != "T1" else ""
                            preset_log_suffix = f"_preset-{preset_name_for_log}" if preset_name_for_log else ""
                            filename = f"{args.subject_id}{space_suffix}{preset_log_suffix}_{name}.stl"
                            runlog["output_files"].append(str(output_dir / filename))
                else:
                    # Check if there is anything to combine
                    if any(m is not None and not m.is_empty for m in surfaces_to_export.values()):
                        space_suffix = f"_space-{args.space}" if args.space != "T1" else ""
                        preset_log_suffix = f"_preset-{preset_name_for_log}" if preset_name_for_log else ""
                        filename = f"{args.subject_id}{space_suffix}{preset_log_suffix}_combined.stl"
                        runlog["output_files"].append(str(output_dir / filename))
                runlog["steps"].append("Export process completed.")

            # --- Cleanup messages ---
            if args.no_clean and not custom_work_dir_path :
                runlog["warnings"].append(f"Temporary folder retained: {work_dir}")
                L.warning(f"Temporary folder retained by --no_clean: {work_dir}")
            elif custom_work_dir_path:
                 L.info(f"Specified work directory {custom_work_dir_path} retained.")
                 if args.no_clean:
                     L.debug("--no_clean is implicit when --work_dir is specified.")


        # Log writing outside the 'with get_work_dir_context_manager'
        log_output_final_location = custom_work_dir_path if custom_work_dir_path else output_dir
        write_log(runlog, str(log_output_final_location), base_name="cortical_surfaces_log")
        L.info("Script finished successfully.")

    except KeyboardInterrupt:
        L.info("\nScript interrupted by user")
        runlog["warnings"].append("Script interrupted by user (KeyboardInterrupt).")
        log_output_final_location = custom_work_dir_path if custom_work_dir_path else output_dir
        write_log(runlog, str(log_output_final_location), base_name="cortical_surfaces_interrupted_log")
        sys.exit(0)
    except Exception as e:
        L.error(f"An unexpected error occurred: {str(e)}", exc_info=args.verbose)
        runlog["warnings"].append(f"An unexpected error occurred: {str(e)}")
        log_output_final_location = custom_work_dir_path if custom_work_dir_path else output_dir
        write_log(runlog, str(log_output_final_location), base_name="cortical_surfaces_error_log")
        sys.exit(1)

if __name__ == "__main__":
    main()

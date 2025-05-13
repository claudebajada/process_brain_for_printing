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
# Import for SGM/Ventricle generation if custom args are used
from .five_tt_utils import run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes, is_vtk_available

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
    parent_parser.add_argument("--work_dir", default=None, help="Optional base directory for intermediate work files. If SGM/Ventricles are requested, 5ttgen outputs will be stored in a '5ttgen_work' subdirectory within this work_dir.")
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
    
    sgm_help_text = (f"Subcortical Gray Matter structures (derived from 5ttgen VTK if space is T1). "
                     f"Use '{const.VTK_KEYWORDS[0]}' for all available, or specific names "
                     f"(e.g., 'L_Thal', 'R_Puta'). Common examples: {', '.join(const.SGM_NAME_EXAMPLES)}. "
                     f"Requires FreeSurfer input for 5ttgen and VTK. Only generated if --space is T1.")
    parser_custom.add_argument("--subcortical-gray", "--subcortical_gray", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=sgm_help_text)
    
    vent_help_text = (f"Ventricular system components (derived from 5ttgen VTK if space is T1). "
                      f"Use '{const.VTK_KEYWORDS[0]}' for all available, or specific names "
                      f"(e.g., '3rd-Ventricle', 'CSF'). Common examples: {', '.join(const.VENT_NAME_EXAMPLES)}. "
                      f"Requires FreeSurfer input for 5ttgen and VTK. Only generated if --space is T1.")
    parser_custom.add_argument("--ventricular-system", "--ventricular_system", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=vent_help_text)
    
    parser_custom.add_argument("--cbm-bs-cc", nargs='*', default=[], choices=const.CBM_BS_CC_CHOICES, metavar='STRUCTURE', help=f"ASEG Cerebellum, BS, CC. Choices: {const.CBM_BS_CC_CHOICES}")
    parser_custom.add_argument("--no-fill-structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip fill for specified CBM/BS/CC structures.")
    parser_custom.add_argument("--no-smooth-structures", nargs='*', choices=const.FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip smooth for specified CBM/BS/CC structures.")
    parser_custom.add_argument('--generate_brain_mask', action='store_true', help="Generate a surface from the brain mask (inflation off, smoothing on by default).")

    # --- Preset Mode Parser ---
    parser_preset = subparsers.add_parser("preset", help="Use presets.", parents=[parent_parser])
    parser_preset.add_argument("--name", required=True, choices=list(PRESETS.keys()), help="Preset name.")

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
             log_output_location_on_fail = custom_work_dir_path if custom_work_dir_path else args.output_dir
             write_log(runlog, str(log_output_location_on_fail), base_name="cortical_surfaces_failed_log")
             sys.exit(1)
        runlog["steps"].append("Initial subject directory validation passed")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        @contextmanager
        def get_work_dir_context_manager(custom_dir, keep_temp, base_out_dir_for_temp):
            if custom_dir: yield str(custom_dir) 
            else:
                with temp_dir("cortical_surf_gen", keep=keep_temp, base_dir=str(base_out_dir_for_temp)) as td_path:
                    yield td_path

        with get_work_dir_context_manager(custom_work_dir_path, args.no_clean, output_dir) as work_dir_str:
            work_dir = Path(work_dir_str) 
            L.info(f"Active work directory: {work_dir}")
            runlog["steps"].append(f"Active work directory: {work_dir}")
            surfaces_to_export: Dict[str, Optional[trimesh.Trimesh]] = {}
            preset_name_for_log = args.name if args.mode == "preset" else "custom"
            preloaded_vtk_meshes: Dict[str, trimesh.Trimesh] = {} 
            needs_5ttgen = False
            if args.mode == "custom":
                sgm_requested = hasattr(args, 'subcortical_gray') and args.subcortical_gray and len(args.subcortical_gray) > 0
                vent_requested = hasattr(args, 'ventricular_system') and args.ventricular_system and len(args.ventricular_system) > 0
                if sgm_requested or vent_requested: needs_5ttgen = True
            
            if needs_5ttgen:
                if args.space.upper() != "T1":
                    L.warning("SGM/Ventricle surfaces from 5ttgen are T1-space only. Requested space: %s. Skipping SGM/Ventricle.", args.space)
                    runlog["warnings"].append("SGM/Ventricle generation skipped (non-T1 space).")
                elif not is_vtk_available():
                    L.warning("VTK unavailable. Skipping 5ttgen SGM/Ventricle generation.")
                    runlog["warnings"].append("SGM/Ventricle generation skipped (VTK unavailable).")
                else:
                    L.info("Attempting 5ttgen for SGM/Ventricle generation (T1 space).")
                    fs_input_dir_for_5ttgen = Path(args.subjects_dir) / "sourcedata" / "freesurfer" / args.subject_id
                    if not fs_input_dir_for_5ttgen.is_dir():
                        L.error(f"Assumed FS input dir for 5ttgen not found: {fs_input_dir_for_5ttgen}. Skipping.")
                        runlog["warnings"].append(f"Assumed FS input dir for 5ttgen not found: {fs_input_dir_for_5ttgen}")
                    else:
                        five_tt_work_dir = work_dir / "5ttgen_processing" / args.subject_id
                        five_tt_work_dir.mkdir(parents=True, exist_ok=True)
                        L.info(f"5ttgen processing directory: {five_tt_work_dir}")
                        
                        # Check if 5ttgen-tmp-* directory already exists
                        existing_5ttgen_tmp_dirs = list(five_tt_work_dir.glob("5ttgen-tmp-*"))
                        run_5ttgen_command = True # Default to running 5ttgen

                        if existing_5ttgen_tmp_dirs:
                            L.info(f"Found existing 5ttgen temporary output in {five_tt_work_dir}: {existing_5ttgen_tmp_dirs[0].name}. Attempting to load VTKs without re-running 5ttgen.")
                            runlog["steps"].append("Skipped 5ttgen execution, found existing tmp directory.")
                            run_5ttgen_command = False # Skip running 5ttgen
                        
                        execute_5ttgen_successful = False
                        if run_5ttgen_command:
                            L.info(f"No existing 5ttgen output found or forcing re-run. Running 5ttgen.")
                            require_cmds(["5ttgen"], logger=L)
                            execute_5ttgen_successful = run_5ttgen_hsvs_save_temp_bids(
                                subject_id=args.subject_id,
                                fs_subject_dir=str(fs_input_dir_for_5ttgen), 
                                subject_work_dir=str(five_tt_work_dir), 
                                session_id=args.session, verbose=args.verbose
                            )
                            if execute_5ttgen_successful:
                                runlog["steps"].append("5ttgen execution completed.")
                            else:
                                L.warning("5ttgen execution failed. Skipping SGM/Ventricle loading.")
                                runlog["warnings"].append("5ttgen execution failed.")
                        
                        # Proceed to load if 5ttgen was skipped (found existing) or ran successfully
                        if not run_5ttgen_command or execute_5ttgen_successful:
                            loaded_vtks = load_subcortical_and_ventricle_meshes(str(five_tt_work_dir))
                            if args.subcortical_gray:
                                if 'all' in args.subcortical_gray:
                                    for k, v in loaded_vtks.items():
                                        if k.startswith("subcortical-"): preloaded_vtk_meshes[k] = v
                                else: 
                                    for name_req in args.subcortical_gray:
                                        for k_loaded, v_loaded in loaded_vtks.items():
                                            if k_loaded.startswith("subcortical-") and name_req in k_loaded:
                                                preloaded_vtk_meshes[k_loaded] = v_loaded
                            if args.ventricular_system:
                                if 'all' in args.ventricular_system:
                                    for k, v in loaded_vtks.items():
                                        if k.startswith("ventricle-"): preloaded_vtk_meshes[k] = v
                                else: 
                                    for name_req in args.ventricular_system:
                                        for k_loaded, v_loaded in loaded_vtks.items():
                                            if k_loaded.startswith("ventricle-") and name_req in k_loaded:
                                                preloaded_vtk_meshes[k_loaded] = v_loaded
                            if preloaded_vtk_meshes:
                                L.info(f"Selected {len(preloaded_vtk_meshes)} SGM/Ventricle meshes: {list(preloaded_vtk_meshes.keys())}")
                                runlog["preloaded_vtk_mesh_keys"] = list(preloaded_vtk_meshes.keys())
                            else: L.info("No SGM/Ventricle meshes matched requests or loaded.")
            
            brain_mask_processed = False
            if (args.mode == "preset" and args.name == "brain_mask_surface") or \
               (args.mode == "custom" and args.generate_brain_mask):
                if args.mode == "preset":
                    L.info("Processing 'brain_mask_surface' preset...")
                    runlog["steps"].append("Selected 'brain_mask_surface' preset")
                    preset_name_for_log = "brain_mask_surface"
                else: 
                    L.info("Processing custom request for brain_mask surface...")
                    runlog["steps"].append("Custom request: brain_mask_surface")
                    preset_name_for_log = "custom_brain_mask"
                brain_mask_mesh = generate_single_brain_mask_surface(
                    args.subjects_dir, args.subject_id, args.space, 0.0, False,
                    args.run, args.session, work_dir, L, args.verbose )
                if brain_mask_mesh and not brain_mask_mesh.is_empty:
                    surfaces_to_export["brain_mask"] = brain_mask_mesh
                    runlog["steps"].append("Generated brain mask surface")
                    L.info("Successfully generated brain mask surface.")
                else:
                    L.warning("Brain mask surface generation failed or resulted in an empty mesh.")
                    runlog["warnings"].append("Brain mask surface generation failed or empty.")
                brain_mask_processed = True

            if not (args.mode == "preset" and args.name == "brain_mask_surface"):
                L.info(f"Parsing standard surface requests for mode: {args.mode}")
                base_cortical_needed, cbm_bs_cc_needed = set(), set() 
                if args.mode == "custom":
                    cortical_req = args.cortical_surfaces if hasattr(args, 'cortical_surfaces') else []
                    cbm_req = args.cbm_bs_cc if hasattr(args, 'cbm_bs_cc') else []
                    if cortical_req or cbm_req: 
                        base_cortical_needed, cbm_bs_cc_needed, _ = parse_custom_surface_args(cortical_req, cbm_req, L)
                    preset_name_for_log = "custom" 
                else:  
                    base_cortical_needed, cbm_bs_cc_needed, _ = parse_preset(args.name)
                    preset_name_for_log = args.name
                runlog["requested_cortical_types"] = sorted(list(base_cortical_needed))
                runlog["requested_cbm_bs_cc"] = sorted(list(cbm_bs_cc_needed))
                if base_cortical_needed or cbm_bs_cc_needed or (args.mode == "custom" and preloaded_vtk_meshes):
                    L.info(f"Base cortical types needed: {base_cortical_needed or 'None'}")
                    L.info(f"CBM/BS/CC structures needed: {cbm_bs_cc_needed or 'None'}")
                    if preloaded_vtk_meshes: L.info(f"Preloaded VTK meshes to include: {list(preloaded_vtk_meshes.keys())}")
                    runlog["steps"].append("Parsed standard surface requests")
                    no_fill = args.no_fill_structures if args.mode == "custom" and hasattr(args, 'no_fill_structures') else []
                    no_smooth = args.no_smooth_structures if args.mode == "custom" and hasattr(args, 'no_smooth_structures') else []
                    generated_standard_surfaces = generate_brain_surfaces(
                        args.subjects_dir, args.subject_id, args.space, tuple(base_cortical_needed),
                        list(cbm_bs_cc_needed), no_fill, no_smooth, args.run, args.session, 
                        args.verbose, str(work_dir), preloaded_vtk_meshes ) 
                    runlog["steps"].append(f"Standard surface generation process completed in {work_dir}")
                    valid_standard_surfaces = {k: v for k, v in generated_standard_surfaces.items() if v is not None and not v.is_empty}
                    surfaces_to_export.update(valid_standard_surfaces)
                elif not brain_mask_processed: 
                     L.info("No surfaces requested for generation (cortical, cbm-bs-cc, or brain_mask).")
                     runlog["steps"].append("No surfaces requested for generation.")

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
                    surfaces_to_export, output_dir, args.subject_id, args.space,
                    preset_name_for_log, args.verbose, args.split_outputs, "stl" )
                if args.split_outputs:
                    for name, mesh_obj in surfaces_to_export.items():
                        if mesh_obj is not None and not mesh_obj.is_empty:
                            space_suffix = f"_space-{args.space}" if args.space != "T1" else ""
                            preset_log_suffix = f"_preset-{preset_name_for_log}" if preset_name_for_log else ""
                            filename = f"{args.subject_id}{space_suffix}{preset_log_suffix}_{name}.stl"
                            runlog["output_files"].append(str(output_dir / filename))
                else:
                    if any(m is not None and not m.is_empty for m in surfaces_to_export.values()):
                        space_suffix = f"_space-{args.space}" if args.space != "T1" else ""
                        preset_log_suffix = f"_preset-{preset_name_for_log}" if preset_name_for_log else ""
                        filename = f"{args.subject_id}{space_suffix}{preset_log_suffix}_combined.stl"
                        runlog["output_files"].append(str(output_dir / filename))
                runlog["steps"].append("Export process completed.")
            if args.no_clean and not custom_work_dir_path :
                runlog["warnings"].append(f"Temporary folder retained: {work_dir}")
                L.warning(f"Temporary folder retained by --no_clean: {work_dir}")
            elif custom_work_dir_path:
                 L.info(f"Specified work directory {custom_work_dir_path} retained.")
        log_output_final_location = custom_work_dir_path if custom_work_dir_path else output_dir
        write_log(runlog, str(log_output_final_location), base_name="cortical_surfaces_log")
        L.info("Script finished successfully.")
    except KeyboardInterrupt: # ... (exception handling as before) ...
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

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
def main() -> None:
    args = _build_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)
    L.setLevel(log_level)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_space = args.space
    is_standard_space = target_space.upper() in ["T1", "MNI"]
    is_subject_space = target_space.startswith("sub-")
    if not is_standard_space and not is_subject_space:
        L.critical(f"Invalid --space: '{target_space}'.")
        sys.exit(1)
    space_mode = "T1" if target_space.upper() == "T1" else ("MNI" if target_space.upper() == "MNI" else "SUBJECT")
    L.info(f"Target space selected: {target_space} (Mode: {space_mode})")

    base_cortical_needed: Set[str] = set()
    cbm_bs_cc_needed: Set[str] = set()
    exact_mesh_keys_requested: List[str] = []
    sgm_request: List[str] = []
    ventricle_request: List[str] = []
    no_fill = getattr(args, 'no_fill_structures', [])
    no_smooth = getattr(args, 'no_smooth_structures', [])

    if args.mode == "custom":
        base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys_requested = parse_custom_surface_args(args.cortical_surfaces, args.cbm_bs_cc)
        sgm_request = args.subcortical_gray
        ventricle_request = args.ventricular_system
    elif args.mode == "preset":
        try:
            base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys_requested = parse_preset(args.name)
            L.info(f"Using preset '{args.name}': Requires Cortical Types={base_cortical_needed}, Other Structures={cbm_bs_cc_needed}, Exact Keys={exact_mesh_keys_requested}")
        except ValueError as e:
            L.critical(e)
            sys.exit(1)
    else:
        L.critical(f"Invalid mode '{args.mode}'")
        sys.exit(1)

    if space_mode != "T1" and "inflated" in base_cortical_needed:
        L.error(f"Inflated surfaces only supported in T1 space (requested: {target_space}).")
        sys.exit(1)

    persistent_5ttgen_path: Optional[Path] = None
    vtk_meshes_loaded: Dict[str, trimesh.Trimesh] = {}
    runlog_vtk_keys: List[str] = []
    vtk_load_needed = bool(sgm_request) or bool(ventricle_request)
    derivatives_dir_path = Path(args.subjects_dir).resolve()

    if vtk_load_needed:
        try:
            if not derivatives_dir_path.is_dir() or derivatives_dir_path.name != 'derivatives':
                L.warning(f"--subjects_dir '{args.subjects_dir}' may not be valid BIDS derivatives.")
            if args.work_dir:
                work_dir_base = Path(args.work_dir).resolve()
            else:
                bids_root_dir = derivatives_dir_path.parent
                work_dir_base = bids_root_dir / "work"
                L.info(f"Defaulting --work_dir to: {work_dir_base}")
            
            bfp_work_base = work_dir_base / "brain_for_printing"
            bfp_work_base.mkdir(parents=True, exist_ok=True)

            subject_label_clean = args.subject_id.replace('sub-', '')
            sub_label = f"sub-{subject_label_clean}"
            ses_label = f"ses-{args.session}" if args.session else None
            
            subject_work_dir_parts = [str(bfp_work_base), sub_label]
            if ses_label:
                subject_work_dir_parts.append(ses_label)
            subject_work_dir = Path(os.path.join(*subject_work_dir_parts))
            subject_work_dir.mkdir(parents=True, exist_ok=True)

            expected_persistent_path = subject_work_dir / "5ttgen_persistent_work"

            if expected_persistent_path.is_dir() and list(expected_persistent_path.glob("5ttgen-tmp-*")):
                L.info(f"Found existing 5ttgen data: {expected_persistent_path}")
                persistent_5ttgen_path = expected_persistent_path
            else:
                L.warning(f"5ttgen data not found or invalid at '{expected_persistent_path}'. Attempting generation...")
                fs_input_dir_options = [
                    derivatives_dir_path / 'sourcedata' / 'freesurfer' / f"sub-{subject_label_clean}",
                    derivatives_dir_path.parent / 'sourcedata' / 'freesurfer' / f"sub-{subject_label_clean}",
                    derivatives_dir_path / 'freesurfer' / f"sub-{subject_label_clean}"
                ]
                fs_input_subject_dir_path = None
                for fs_path in fs_input_dir_options:
                    if fs_path.is_dir():
                        fs_input_subject_dir_path = fs_path
                        L.info(f"Found FreeSurfer input directory: {fs_path}")
                        break
                
                if not fs_input_subject_dir_path:
                    L.error("Could not find FreeSurfer input directory for 5ttgen. Cannot generate VTK meshes.")
                    persistent_5ttgen_path = None
                else:
                    generated_path_str = run_5ttgen_hsvs_save_temp_bids(
                        subject_id=subject_label_clean, # Pass only the label
                        fs_subject_dir=str(fs_input_subject_dir_path),
                        subject_work_dir=str(subject_work_dir), 
                        session_id=args.session # Pass session_id if available
                    )
                    if generated_path_str and Path(generated_path_str).is_dir():
                        L.info(f"5ttgen generation successful: {generated_path_str}")
                        persistent_5ttgen_path = Path(generated_path_str)
                    else:
                        L.error("Failed 5ttgen generation.")
                        persistent_5ttgen_path = None
        except Exception as e:
            L.error(f"Error during 5ttgen handling: {e}", exc_info=args.verbose)
            L.warning("Continuing without 5ttgen/VTK meshes due to error.")
            persistent_5ttgen_path = None

        if persistent_5ttgen_path and persistent_5ttgen_path.is_dir():
            vtk_meshes_loaded_all = load_subcortical_and_ventricle_meshes(str(persistent_5ttgen_path))
            available_vtk_keys = set(vtk_meshes_loaded_all.keys())
            target_vtk_keys: Set[str] = set()
            
            sgm_req_keywords = {k for k in sgm_request if k == const.VTK_KEYWORDS[0]}
            sgm_req_specific = {k for k in sgm_request if k != const.VTK_KEYWORDS[0]}
            
            if const.VTK_KEYWORDS[0] in sgm_req_keywords:
                target_vtk_keys.update(k for k in available_vtk_keys if k.startswith('subcortical-'))
            for name_req in sgm_req_specific:
                potential_key = f"subcortical-{name_req}"
                if potential_key in available_vtk_keys:
                    target_vtk_keys.add(potential_key)
                    L.info(f"Found requested SGM: '{name_req}' as '{potential_key}'")
                else:
                    L.warning(f"Requested SGM '{name_req}' (as '{potential_key}') not found in VTK outputs.")
            
            vent_req_keywords = {k for k in ventricle_request if k == const.VTK_KEYWORDS[0]}
            vent_req_specific = {k for k in ventricle_request if k != const.VTK_KEYWORDS[0]}

            if const.VTK_KEYWORDS[0] in vent_req_keywords:
                target_vtk_keys.update(k for k in available_vtk_keys if k.startswith('ventricle-') or k.startswith('vessel-'))
            for name_req in vent_req_specific:
                found_vent_vessel = False
                potential_key_vessel = f"vessel-{name_req}"
                potential_key_vent = f"ventricle-{name_req}"
                if potential_key_vessel in available_vtk_keys:
                    target_vtk_keys.add(potential_key_vessel)
                    found_vent_vessel = True
                    L.info(f"Found requested Vent/Vessel: '{name_req}' as '{potential_key_vessel}'")
                elif potential_key_vent in available_vtk_keys:
                    target_vtk_keys.add(potential_key_vent)
                    found_vent_vessel = True
                    L.info(f"Found requested Vent/Vessel: '{name_req}' as '{potential_key_vent}'")
                if not found_vent_vessel:
                    L.warning(f"Requested Vent/Vessel '{name_req}' not found in VTK outputs.")
            
            vtk_meshes_loaded = {k: vtk_meshes_loaded_all[k] for k in target_vtk_keys if k in vtk_meshes_loaded_all}
            exact_mesh_keys_requested.extend(sorted(list(vtk_meshes_loaded.keys())))
            exact_mesh_keys_requested = sorted(list(set(exact_mesh_keys_requested))) 
            runlog_vtk_keys = sorted(list(vtk_meshes_loaded.keys()))
            L.info(f"Final VTK meshes selected for loading: {runlog_vtk_keys}")
        else:
            L.warning("No valid 5ttgen directory available for VTK loading. Skipping VTK meshes.")
            vtk_meshes_loaded = {}
            runlog_vtk_keys = []
    else:
        L.info("VTK loading not requested.")
        vtk_meshes_loaded = {}
        runlog_vtk_keys = []

    if not exact_mesh_keys_requested:
        L.info("No surfaces specified or loaded after all processing steps. Nothing to do.")
        # write_log before exiting if you want to capture this state, and then exit.
        # For now, let it proceed to tool check and temp_dir context, it might exit there if no tools needed / no meshes.
    
    required_tools = []
    if cbm_bs_cc_needed:
        required_tools.append("mri_binarize")

    if vtk_load_needed and persistent_5ttgen_path is None:
        required_tools.append("5ttgen")

    if space_mode != "T1":
        required_tools.extend(["antsApplyTransforms", "warpinit", "mrcat"])
    
    if required_tools:
        try:
            require_cmds(list(set(required_tools)), logger=L) # Remove duplicates
        except SystemExit: 
            L.critical("Required external tools missing. Please install them and ensure they are in your PATH.")
            _log_req_cortical = args.cortical_surfaces if args.mode == 'custom' else PRESETS.get(args.name, [])
            _log_req_cbm_bs_cc = args.cbm_bs_cc if args.mode == 'custom' else PRESETS.get(args.name, [])

            temp_runlog_on_error = {
                "tool": "brain_for_printing_surfaces", "mode": args.mode, "subject_id": args.subject_id,
                "space": target_space, "space_mode": space_mode,
                "req_cortical": _log_req_cortical, 
                "req_cbm_bs_cc": _log_req_cbm_bs_cc, 
                 "parsed_cortical_types": sorted(list(base_cortical_needed)), 
                 "parsed_cbm_bs_cc": sorted(list(cbm_bs_cc_needed)), 
                "vtk_load_attempted": vtk_load_needed,
                "5ttgen_persistent_work_dir_used": str(persistent_5ttgen_path) if persistent_5ttgen_path else None,
                "loaded_vtk_keys": runlog_vtk_keys, 
                "final_mesh_keys_requested": exact_mesh_keys_requested, 
                 "warnings": [f"Exited due to missing tools: {required_tools}"], "output_files": [], "steps": ["Tool check failed"],
            }
            if args.mode == "preset": temp_runlog_on_error["preset_name"] = args.name
            write_log(temp_runlog_on_error, out_dir, base_name=f"surfaces_{args.mode}_critical_error_log")
            sys.exit(1)

    log_req_cortical_items = []
    log_req_cbm_bs_cc_items = []

    if args.mode == 'custom':
        log_req_cortical_items = args.cortical_surfaces
        log_req_cbm_bs_cc_items = args.cbm_bs_cc
    else:  # preset mode
        preset_list_items = PRESETS.get(args.name, [])
        for item in preset_list_items:
            item_base_type = item.split('-', 1)[-1] if '-' in item else item
            if item in const.CORTICAL_CHOICES or item_base_type in const.CORTICAL_TYPES:
                log_req_cortical_items.append(item)
            if item in const.CBM_BS_CC_CHOICES:
                log_req_cbm_bs_cc_items.append(item)

    runlog = {
        "tool": "brain_for_printing_surfaces", "mode": args.mode, "subject_id": args.subject_id,
        "space": target_space, "space_mode": space_mode,
        "req_cortical": log_req_cortical_items, 
        "req_cbm_bs_cc": log_req_cbm_bs_cc_items, 
        "req_sgm": sgm_request, "req_ventricles": ventricle_request,
        "parsed_cortical_types": sorted(list(base_cortical_needed)),
        "parsed_cbm_bs_cc": sorted(list(cbm_bs_cc_needed)),
        "vtk_load_attempted": vtk_load_needed,
        "5ttgen_persistent_work_dir_used": str(persistent_5ttgen_path) if persistent_5ttgen_path else None,
        "loaded_vtk_keys": runlog_vtk_keys, 
        "final_mesh_keys_requested": exact_mesh_keys_requested, 
        "no_fill_structures": no_fill, "no_smooth_structures": no_smooth,
        "split_outputs": args.split_outputs, "output_dir": str(out_dir),
        "steps": [], "warnings": [], "output_files": [],
    }
    if args.mode == "preset":
        runlog["preset_name"] = args.name
    
    final_meshes: Dict[str, Optional[trimesh.Trimesh]] = {}
    temp_tag = f"surfgen_{space_mode}_{target_space}" if space_mode=="SUBJECT" else f"surfgen_{space_mode}"
    
    with temp_dir(temp_tag, keep=args.no_clean, base_dir=out_dir) as tmp_dir_str:
        tmp_dir_path = Path(tmp_dir_str)
        L.info(f"Using temporary folder for generation/warping: {tmp_dir_path}")
        runlog["steps"].append(f"Created temp dir => {tmp_dir_path}")

        generate_surfaces_cortical_types = tuple(base_cortical_needed)
        generate_surfaces_extract_structures = list(cbm_bs_cc_needed)
        
        preloaded_meshes_for_generator = {}
        if space_mode == "T1": 
            preloaded_meshes_for_generator.update(vtk_meshes_loaded) 
        
        if generate_surfaces_cortical_types or generate_surfaces_extract_structures:
            L.info(f"Calling generate_brain_surfaces for cortical types: {generate_surfaces_cortical_types} and other structures: {generate_surfaces_extract_structures}")
            if space_mode == "T1" and preloaded_meshes_for_generator:
                 L.info(f"Preloading {len(preloaded_meshes_for_generator)} VTK meshes into generate_brain_surfaces for T1 space.")

            try:
                generated_standard_meshes = generate_brain_surfaces(
                    subjects_dir=str(derivatives_dir_path), subject_id=args.subject_id,
                    space=target_space, surfaces=generate_surfaces_cortical_types,
                    extract_structures=generate_surfaces_extract_structures,
                    no_fill_structures=no_fill, no_smooth_structures=no_smooth,
                    run=args.run, session=args.session, verbose=args.verbose, tmp_dir=str(tmp_dir_path),
                    preloaded_vtk_meshes=preloaded_meshes_for_generator 
                )
                final_meshes.update(generated_standard_meshes) 
                runlog["steps"].append(f"Surface generation via generate_brain_surfaces completed for space '{target_space}'. Found {len(generated_standard_meshes)} meshes.")
            except (ValueError, FileNotFoundError, Exception) as e:
                L.error(f"generate_brain_surfaces call failed: {e}", exc_info=args.verbose)
                runlog["warnings"].append(f"generate_brain_surfaces call failed: {e}")
                if not vtk_meshes_loaded or space_mode != "T1": 
                     write_log(runlog, out_dir, base_name=f"surfaces_{args.mode}_critical_error_log")
                     sys.exit(1)
        elif space_mode == "T1" and vtk_meshes_loaded: 
             L.info("Only VTK meshes requested in T1 space. Adding them directly.")
             final_meshes.update(vtk_meshes_loaded)
             runlog["steps"].append(f"Added {len(vtk_meshes_loaded)} preloaded VTK meshes directly for T1 space.")

        if space_mode != "T1" and vtk_load_needed:
            vtk_keys_in_final_requested = [k for k in runlog_vtk_keys if k in exact_mesh_keys_requested]
            if vtk_keys_in_final_requested:
                L.warning(f"Warping of loaded VTK meshes ({vtk_keys_in_final_requested}) to space '{target_space}' is not handled by this script's generate_brain_surfaces call for non-T1 space. They will be excluded if they were part of 'exact_mesh_keys_requested'.")
                runlog["warnings"].append(f"Warping of VTK meshes to {target_space} not implemented; excluded: {', '.join(vtk_keys_in_final_requested)}.")
                for k_remove in vtk_keys_in_final_requested:
                    final_meshes.pop(k_remove, None)
                exact_mesh_keys_requested = [k for k in exact_mesh_keys_requested if k not in vtk_keys_in_final_requested]
                runlog["final_mesh_keys_requested"] = exact_mesh_keys_requested 

        meshes_to_export: Dict[str, trimesh.Trimesh] = {}
        if not exact_mesh_keys_requested: 
            L.warning("After processing, no mesh keys remain in the request list. Nothing to export.")
        else:
            L.info(f"Filtering for final export based on updated keys: {exact_mesh_keys_requested}")
            for key in exact_mesh_keys_requested:
                mesh_obj = final_meshes.get(key)
                if isinstance(mesh_obj, trimesh.Trimesh) and not mesh_obj.is_empty:
                    meshes_to_export[key] = mesh_obj
                else:
                    L.warning(f"Requested key '{key}' not available, empty, or not a Trimesh object in final_meshes dict. Skipping for export.")
                    runlog["warnings"].append(f"Skipped export of unavailable/invalid key: {key}")

        if not meshes_to_export:
            L.error("No meshes available for export after all processing and filtering. Exiting.")
            runlog["warnings"].append("No meshes available for export.")
            # Define out_space_tag here if not defined yet to avoid NameError in write_log
            out_space_tag_local = locals().get('out_space_tag', target_space.upper() if is_standard_space else target_space)
            write_log(runlog, out_dir, base_name=f"surfaces_{args.mode}_space-{out_space_tag_local}_empty_export_log")
            sys.exit(1)
            
        out_space_tag = target_space if is_subject_space else target_space.upper()
        if args.split_outputs:
            L.info(f"Exporting {len(meshes_to_export)} meshes separately...")
            subject_label_clean = args.subject_id.replace('sub-', '')
            for name, mesh_to_export_obj in meshes_to_export.items(): 
                bids_like_name_parts = [f"sub-{subject_label_clean}"]
                if args.session: bids_like_name_parts.append(f"ses-{args.session}")
                if args.run: bids_like_name_parts.append(f"run-{args.run}")
                bids_like_name_parts.append(f"space-{out_space_tag}")
                
                if name.endswith("_L"): bids_like_name_parts.append(f"hemi-L_label-{name[:-2]}_surf")
                elif name.endswith("_R"): bids_like_name_parts.append(f"hemi-R_label-{name[:-2]}_surf")
                else: bids_like_name_parts.append(f"label-{name}_mesh")

                fname = "_".join(bids_like_name_parts) + ".stl"
                out_path = out_dir / fname
                try:
                    L.info(f"Exporting '{name}' to {out_path}...")
                    mesh_to_export_obj.export(out_path, file_type="stl")
                    runlog["steps"].append(f"Exported {name} => {out_path}")
                    runlog["output_files"].append(str(out_path))
                except Exception as e:
                    L.error(f"Failed to export {name} to {out_path}: {e}", exc_info=args.verbose)
                    runlog["warnings"].append(f"Export failed for {name}: {e}")
        else: 
            L.info(f"Merging {len(meshes_to_export)} meshes for combined output...")
            mesh_list_to_combine = [m for m in meshes_to_export.values() if m is not None and not m.is_empty] 
            if not mesh_list_to_combine:
                L.error("No valid meshes to merge for combined output.")
                runlog["warnings"].append("Merge skipped: No valid meshes found for concatenation.")
            else:
                try:
                    combined_mesh_output = trimesh.util.concatenate(mesh_list_to_combine) 
                    L.info(f"Successfully concatenated {len(mesh_list_to_combine)} meshes.")
                    runlog["steps"].append(f"Concatenated {len(mesh_list_to_combine)} meshes")
                except Exception as e:
                    L.error(f"Mesh concatenation failed: {e}", exc_info=args.verbose)
                    runlog["warnings"].append(f"Concatenation failed: {e}")
                    combined_mesh_output = None
                
                if combined_mesh_output and not combined_mesh_output.is_empty:
                    subject_label_clean = args.subject_id.replace('sub-', '')
                    fname_parts = [f"sub-{subject_label_clean}"]
                    if args.session: fname_parts.append(f"ses-{args.session}")
                    if args.run: fname_parts.append(f"run-{args.run}")
                    fname_parts.append(f"space-{out_space_tag}")
                    desc_suffix = f"{args.mode}"
                    if args.mode == 'preset': desc_suffix += f"-{args.name}" 
                    combined_desc = f"desc-combined-{desc_suffix}_meshes"
                    fname_parts.append(combined_desc)
                    
                    out_fname = "_".join(fname_parts) + ".stl"
                    out_path = out_dir / out_fname
                    try:
                        L.info(f"Exporting combined mesh to {out_path}...")
                        combined_mesh_output.export(out_path, file_type="stl")
                        runlog["steps"].append(f"Exported merged mesh => {out_path}")
                        runlog["output_files"].append(str(out_path))
                    except Exception as e:
                        L.error(f"Failed to export combined mesh to {out_path}: {e}", exc_info=args.verbose)
                        runlog["warnings"].append(f"Export failed for combined mesh: {e}")
                elif combined_mesh_output is None: 
                    L.warning("Combined mesh creation failed. Skipping export of combined mesh.")
                else: 
                    L.warning("Combined mesh is empty after concatenation. Skipping export.")
                    runlog["warnings"].append("Combined mesh was empty; not exported.")

        if args.no_clean: 
            if 'tmp_dir_path' in locals() and tmp_dir_path.exists():
                runlog["warnings"].append(f"Temporary folder kept: {tmp_dir_path}")
                L.info(f"Temporary folder kept: {tmp_dir_path}")
            else:
                runlog["warnings"].append(f"Temporary folder was not created or already removed, --no_clean has no effect.")

    if 'out_space_tag' not in locals(): 
        out_space_tag = target_space.upper() if is_standard_space else target_space

    log_base_name_parts = ["surfaces", args.mode]
    if args.mode == 'preset' and args.name: 
        log_base_name_parts.append(args.name)
    log_base_name_parts.append(f"space-{out_space_tag}")
    log_base = "_".join(log_base_name_parts)
    
    try:
        write_log(runlog, out_dir, base_name=log_base + "_log")
    except Exception as e:
        L.error(f"Failed to write final JSON log: {e}")

    L.info("Script finished.")

#!/usr/bin/env python
# brain_for_printing/cli_cortical_surfaces.py
#
# CLI for generating brain surfaces from different categories:
# 1. Cortical surfaces (pial, white, mid, inflated)
# 2. Subcortical Gray Matter (from 5ttgen VTK)
# 3. Cerebellum, Brainstem, Corpus Callosum (from ASEG)
# 4. Ventricular System (from 5ttgen VTK)
# Supports custom specification via category flags or predefined presets (presets cover 1 & 3).
# Manages 5ttgen data generation/access via a dedicated work directory.

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
import vtk # Keep vtk imports if _read_vtk_polydata etc. are used internally by surfaces.py funcs
from vtk.util import numpy_support

# Assuming local imports are correct
from .io_utils import temp_dir, require_cmds, flexible_match #
from .log_utils import get_logger, write_log #
from .surfaces import generate_brain_surfaces, run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes # Updated imports
from . import constants as const #

# --- Constants for Argument Choices ---
CORTICAL_TYPES = ["pial", "white", "mid", "inflated"] # Added inflated
HEMI_CORTICAL_TYPES = [f"{h}-{t}" for h in ["lh", "rh"] for t in CORTICAL_TYPES]
CORTICAL_CHOICES = CORTICAL_TYPES + HEMI_CORTICAL_TYPES

# ASEG-derived structures for --cbm-bs-cc flag
CBM_BS_CC_CHOICES = [
    "brainstem", "cerebellum_wm", "cerebellum_cortex",
    "cerebellum", "corpus_callosum"
]
OTHER_SURFACE_CHOICES = CBM_BS_CC_CHOICES # Alias

# Keywords for VTK-derived structures ('all' is the only keyword now)
VTK_KEYWORDS = ["all"]

# Example names for VTK-derived subcortical structures
SGM_NAME_EXAMPLES = [
    'L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 'L_Thal',
    'R_Accu', 'R_Amyg', 'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 'R_Thal'
]

# Example names for VTK-derived ventricular/vessel structures
VENT_NAME_EXAMPLES = [
    '3rd-Ventricle', '4th-Ventricle',
    'Left-Inf-Lat-Vent', 'Left_LatVent_ChorPlex',
    'Right-Inf-Lat-Vent', 'Right_LatVent_ChorPlex'
]

# Presets
PRESETS = {
    "pial_brain": ['lh-pial', 'rh-pial', 'corpus_callosum', 'cerebellum', 'brainstem'],
    "white_brain": ['lh-white', 'rh-white', 'corpus_callosum', 'cerebellum_wm', 'brainstem'],
    "mid_brain": ['lh-mid', 'rh-mid', 'corpus_callosum', 'cerebellum', 'brainstem'],
    "cortical_pial": ['lh-pial', 'corpus_callosum', 'rh-pial'],
    "cortical_white": ['lh-white', 'corpus_callosum', 'rh-white'],
    "cortical_mid": ['lh-mid', 'corpus_callosum', 'rh-mid'],
}


# Structures that might have fill/smooth options applied (CBM_BS_CC choices)
FILL_SMOOTH_CHOICES = CBM_BS_CC_CHOICES

# Define logger at module level
L = logging.getLogger("brain_for_printing_surfaces")

# --------------------------------------------------------------------------- #
# Helper Functions (Parsing logic remains similar, VTK helpers moved to surfaces.py)
# --------------------------------------------------------------------------- #

def parse_cortical_and_cbmbscc_args(
    cortical_request: List[str],
    cbm_bs_cc_request: List[str]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Parses --cortical-surfaces and --cbm-bs-cc args. Handles 'inflated'.
    Returns: (base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys)
    """
    base_cortical_needed: Set[str] = set()
    cbm_bs_cc_needed: Set[str] = set()
    exact_mesh_keys: Set[str] = set()
    # Process cortical requests (including inflated)
    for req_surf in cortical_request:
        if req_surf in CORTICAL_TYPES:
            base_cortical_needed.add(req_surf)
            exact_mesh_keys.add(f"{req_surf}_L")
            exact_mesh_keys.add(f"{req_surf}_R")
        elif req_surf in HEMI_CORTICAL_TYPES:
            try:
                hemi_prefix, base_type = req_surf.split('-', 1);
                # --- Indentation Fix Applied Below ---
                if hemi_prefix in ['lh', 'rh'] and base_type in CORTICAL_TYPES:
                    base_cortical_needed.add(base_type)
                    suffix = "_L" if hemi_prefix == 'lh' else "_R"
                    exact_mesh_keys.add(f"{base_type}{suffix}")
                else:
                    L.warning(f"Ignoring malformed hemi surf: {req_surf}")
            except ValueError:
                L.warning(f"Ignoring malformed hemi surf: {req_surf}")
        else:
            L.warning(f"Ignoring unrecognized cortical surf: {req_surf}")
    # Process CBM/BS/CC requests
    for req_other in cbm_bs_cc_request:
        if req_other in CBM_BS_CC_CHOICES:
            cbm_bs_cc_needed.add(req_other)
            exact_mesh_keys.add(req_other)
        else: L.warning(f"Ignoring unrecognized cbm-bs-cc surf: {req_other}")
    return base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys

# --------------------------------------------------------------------------- #
# CLI Argument Parser Setup
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate brain surfaces.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parent_parser = argparse.ArgumentParser(add_help=False) # Common arguments
    parent_parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory (e.g., /path/to/bids/derivatives).")
    # --- MODIFIED: Added --work_dir ---
    parent_parser.add_argument("--work_dir", default=None, help="Optional base directory for intermediate work files. Defaults to a 'work' directory alongside 'derivatives'.")
    parent_parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-CB).")
    parent_parser.add_argument("--output_dir", default=".", help="Output directory for final STL/OBJ files.")
    parent_parser.add_argument("--space", choices=["T1", "MNI"], default="T1", help="Output space (MNI invalid for inflated).")
    parent_parser.add_argument("--run", default=None, help="BIDS run entity.")
    parent_parser.add_argument("--session", default=None, help="BIDS session entity.")
    parent_parser.add_argument("--split_outputs", action="store_true", help="Export surfaces separately.")
    parent_parser.add_argument("--out_warp", default="warp.nii", help="Warp filename (if MNI space, used in temp dir).")
    parent_parser.add_argument("--no_clean", action="store_true", help="Keep temporary folder used for ASEG/Cortical generation.")
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")

    subparsers = parser.add_subparsers(dest="mode", required=True, title="Modes")

    # Custom Mode Parser
    parser_custom = subparsers.add_parser("custom", help="Specify surfaces manually.", parents=[parent_parser])
    parser_custom.add_argument("--cortical-surfaces", "--cortical_surfaces", nargs='*', default=[], choices=CORTICAL_CHOICES, metavar='SURFACE', help=f"Cortical surfaces (inc. inflated). Choices: {CORTICAL_CHOICES}")
    sgm_help_text = (
        f"VTK Subcortical Gray. Use '{VTK_KEYWORDS[0]}' to include all found structures, "
        f"or specific names derived from 5ttgen filenames (e.g., 'L_Thal', 'R_Puta'). "
        f"Common examples based on abbreviations: {', '.join(SGM_NAME_EXAMPLES)}. "
        f"Exact names depend on the VTK files in the 5ttgen work directory."
    )    
    parser_custom.add_argument("--subcortical-gray", "--subcortical_gray", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=sgm_help_text)
    parser_custom.add_argument("--cbm-bs-cc", nargs='*', default=[], choices=CBM_BS_CC_CHOICES, metavar='STRUCTURE', help=f"ASEG Cerebellum, BS, CC. Choices: {CBM_BS_CC_CHOICES}")
    vent_help_text = (
        f"VTK Ventricles/Vessels. Use '{VTK_KEYWORDS[0]}' to include all found structures, "
        f"or specific names derived from 5ttgen VTK filenames (e.g., '3rd-Ventricle.vtk' -> '3rd-Ventricle'; "
        f"excludes *_init.vtk, *_first*.vtk). Common examples: {', '.join(VENT_NAME_EXAMPLES)}. "
        f"Exact names depend on the VTK files in the 5ttgen work directory."
    )    
    parser_custom.add_argument("--ventricular-system", "--ventricular_system", nargs='*', default=[], metavar='KEYWORD_OR_NAME', help=vent_help_text)
    parser_custom.add_argument("--no-fill-structures", "--no_fill_structures", nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip fill for CBM/BS/CC.")
    parser_custom.add_argument("--no-smooth-structures", "--no_smooth_structures", nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Skip smooth for CBM/BS/CC.")

    # Preset Mode Parser
    parser_preset = subparsers.add_parser("preset", help="Use presets (cortical & CBM/BS/CC).", parents=[parent_parser])
    parser_preset.add_argument("--name", required=True, choices=list(PRESETS.keys()), help="Preset name.")
    parser_preset.add_argument("--no-fill-structures", "--no_fill_structures", nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Preset override: skip fill.")
    parser_preset.add_argument("--no-smooth-structures", "--no_smooth_structures", nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE', help="Preset override: skip smooth.")

    return parser

# --------------------------------------------------------------------------- #
# Main Execution Logic
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _build_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)
    L.setLevel(log_level)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize Request Variables ---
    cortical_request: List[str] = []; cbm_bs_cc_request: List[str] = []
    sgm_request: List[str] = []; ventricle_request: List[str] = []
    no_fill = getattr(args, 'no_fill_structures', [])
    no_smooth = getattr(args, 'no_smooth_structures', [])

    # --- Determine Requests based on Mode ---
    if args.mode == "custom":
        cortical_request = args.cortical_surfaces; cbm_bs_cc_request = args.cbm_bs_cc
        sgm_request = args.subcortical_gray; ventricle_request = args.ventricular_system
    elif args.mode == "preset":
        preset_surfaces = PRESETS[args.name]; L.info(f"Using preset '{args.name}': {', '.join(preset_surfaces)}")
        for item in preset_surfaces:
            if item in CORTICAL_CHOICES: cortical_request.append(item)
            elif item in CBM_BS_CC_CHOICES: cbm_bs_cc_request.append(item)
            else: L.warning(f"Preset item '{item}' not recognized.")
    else: L.critical(f"Invalid mode '{args.mode}'"); sys.exit(1)

    # --- Check for invalid MNI + Inflated request early ---
    if args.space.upper() == "MNI":
        if any("inflated" in s for s in cortical_request):
            L.error("Cannot request inflated surfaces when `space` is MNI. Use `--space T1`."); sys.exit(1)

    # --- Parse Cortical and CBM/BS/CC requests ---
    base_cortical_needed, cbm_bs_cc_needed, exact_mesh_keys_requested = parse_cortical_and_cbmbscc_args(cortical_request, cbm_bs_cc_request)

    # --- Handle VTK Request / 5ttgen (MODIFIED for Work Directory) ---
    persistent_5ttgen_path: Optional[Path] = None # Path to the persistent 5ttgen data dir within work_dir
    vtk_meshes_loaded: Dict[str, trimesh.Trimesh] = {} # Meshes loaded from VTK
    runlog_vtk_keys: List[str] = [] # Keys of loaded VTK meshes for logging
    vtk_load_needed = bool(sgm_request) or bool(ventricle_request) # Is VTK loading required?

    if vtk_load_needed:
        L.info(f"VTK structures requested: SGM={sgm_request}, Ventricles={ventricle_request}")
        try:
            # --- Define Paths (Derivatives, Work, Subject-Specific Work) ---
            derivatives_dir_path = Path(args.subjects_dir).resolve() # Resolve to absolute path
            if not derivatives_dir_path.is_dir() or derivatives_dir_path.name != 'derivatives':
                 L.warning(f"Provided --subjects_dir '{args.subjects_dir}' does not seem to end in 'derivatives'. Assuming it's the correct base for finding FreeSurfer inputs.")
                 # We might need derivatives_dir_path later if generate_brain_surfaces expects it.
                 # Let's assume it's correct for now, but the default work_dir logic might be affected.

            # Determine Base Work Directory
            if args.work_dir:
                work_dir_base = Path(args.work_dir).resolve()
            else:
                # Default: ../work relative to derivatives_dir_path parent (BIDS root)
                bids_root_dir = derivatives_dir_path.parent
                work_dir_base = bids_root_dir / "work"
                L.info(f"--work_dir not specified, defaulting to: {work_dir_base}")

            # Create Base Work Directory Structure if it doesn't exist
            bfp_work_base = work_dir_base / "brain_for_printing"
            bfp_work_base.mkdir(parents=True, exist_ok=True)
            L.info(f"Using BrainForPrinting work base: {bfp_work_base}")

            # Determine Subject-Specific Work Directory
            subject_label_clean = args.subject_id.replace('sub-', '')
            sub_label = f"sub-{subject_label_clean}"
            ses_label = f"ses-{args.session}" if args.session else None
            subject_work_dir_parts = [str(bfp_work_base), sub_label]
            if ses_label: subject_work_dir_parts.append(ses_label)
            subject_work_dir = Path(os.path.join(*subject_work_dir_parts))
            subject_work_dir.mkdir(parents=True, exist_ok=True) # Ensure this specific dir exists too
            L.info(f"Subject-specific work directory: {subject_work_dir}")

            # --- Check for existing persistent 5ttgen data in the work directory ---
            expected_persistent_path = subject_work_dir / "5ttgen_persistent_work"
            if expected_persistent_path.is_dir():
                # Check if the crucial nested '5ttgen-tmp-*' directory exists inside
                nested_check = list(expected_persistent_path.glob("5ttgen-tmp-*"))
                if nested_check:
                     L.info(f"Found existing and seemingly valid 5ttgen data in work directory: {expected_persistent_path}. Using this data.")
                     persistent_5ttgen_path = expected_persistent_path # Set path to use existing data
                else:
                     L.warning(f"Found existing directory at {expected_persistent_path}, but it lacks the expected '5ttgen-tmp-*' subdirectory. Will attempt regeneration.")
                     persistent_5ttgen_path = None # Force regeneration
            else:
                # If not found, attempt generation into the work directory
                L.warning(f"Persistent 5ttgen data not found at: '{expected_persistent_path}'. Attempting generation...")
                persistent_5ttgen_path = None # Flag that generation is needed

            # --- Generate 5ttgen data if needed ---
            if persistent_5ttgen_path is None: # Only run if existing data wasn't found or was invalid
                 # Find FreeSurfer input dir (relative to derivatives or sourcedata)
                 # Prioritize derivatives/sourcedata/freesurfer if exists, else assume it might be under sourcedata/freesurfer or derivatives/freesurfer
                 fs_input_dir_options = [
                     derivatives_dir_path / 'sourcedata' / 'freesurfer' / f"sub-{subject_label_clean}",
                     derivatives_dir_path.parent / 'sourcedata' / 'freesurfer' / f"sub-{subject_label_clean}",
                     derivatives_dir_path / 'freesurfer' / f"sub-{subject_label_clean}" # Less standard, but maybe
                 ]
                 fs_input_subject_dir_path = None
                 for fs_path in fs_input_dir_options:
                     if fs_path.is_dir():
                         fs_input_subject_dir_path = fs_path
                         break

                 if not fs_input_subject_dir_path:
                     L.error(f"Could not find FreeSurfer input subject directory at expected locations:")
                     for fs_path in fs_input_dir_options: L.error(f" - {fs_path}")
                     L.error("Please ensure FreeSurfer outputs exist in sourcedata/freesurfer/ or derivatives/freesurfer/.")
                     sys.exit(1)
                 L.info(f"Using FreeSurfer input directory: {fs_input_subject_dir_path}")

                 # Call the modified generation function, passing the subject_work_dir
                 generated_path_str = run_5ttgen_hsvs_save_temp_bids(
                     subject_id=subject_label_clean, # Pass cleaned ID
                     fs_subject_dir=str(fs_input_subject_dir_path),
                     subject_work_dir=subject_work_dir, # Pass the dedicated work path
                     session_id=args.session
                     # Removed bids_root_dir and pipeline_name
                 )

                 # Check the result of the generation attempt
                 if generated_path_str and Path(generated_path_str).is_dir():
                     L.info(f"5ttgen generation successful, persistent output at: {generated_path_str}")
                     persistent_5ttgen_path = Path(generated_path_str) # Use the path returned by the function
                 else:
                     L.error("Failed 5ttgen generation or output directory could not be confirmed.")
                     persistent_5ttgen_path = None # Generation failed

        except Exception as e:
            L.error(f"Error during work directory setup or 5ttgen generation: {e}")
            import traceback
            traceback.print_exc() # Print traceback for more details
            sys.exit(1) # Exit if path handling fails critically

        # --- Load VTK meshes (using persistent_5ttgen_path determined above) ---
        if persistent_5ttgen_path and persistent_5ttgen_path.is_dir():
             # Call the load function, passing the path to the persistent directory
             vtk_meshes_loaded_all = load_subcortical_and_ventricle_meshes(persistent_5ttgen_path)
             available_vtk_keys = set(vtk_meshes_loaded_all.keys())
             target_vtk_keys: Set[str] = set() # Store the keys of VTK meshes we actually want

             # Filter SGM based on sgm_request (logic remains the same)
             sgm_req_keywords = {k for k in sgm_request if k=='all'}
             sgm_req_specific = {k for k in sgm_request if k!='all'}
             if 'all' in sgm_req_keywords:
                 target_vtk_keys.update(k for k in available_vtk_keys if k.startswith('subcortical-'))
             for name_req in sgm_req_specific:
                 potential_key = f"subcortical-{name_req}"
                 if potential_key in available_vtk_keys:
                     target_vtk_keys.add(potential_key)
                 else:
                     L.warning(f"Requested SGM name '{name_req}' (key: {potential_key}) not found among loaded VTK files.")

             # Filter Ventricles based on ventricle_request (logic remains the same)
             vent_req_keywords = {k for k in ventricle_request if k=='all'}
             vent_req_specific = {k for k in ventricle_request if k!='all'}
             if 'all' in vent_req_keywords:
                 target_vtk_keys.update(k for k in available_vtk_keys if k.startswith('ventricle-') or k.startswith('vessel-'))
             for name_req in vent_req_specific:
                  found=False
                  for prefix in ["ventricle-","vessel-"]:
                       potential_key=f"{prefix}{name_req}"
                       if potential_key in available_vtk_keys:
                           target_vtk_keys.add(potential_key)
                           found=True
                           L.debug(f"Mapped '{name_req}' to '{potential_key}'")
                           break # Found it for this name_req
                  if not found:
                      L.warning(f"Requested ventricular/vessel name '{name_req}' not found among loaded VTK files.")

             # Finalize VTK meshes to include based on filtered keys
             vtk_meshes_loaded = {k: vtk_meshes_loaded_all[k] for k in target_vtk_keys if k in vtk_meshes_loaded_all}
             exact_mesh_keys_requested.update(vtk_meshes_loaded.keys()) # Add selected VTK keys to overall request
             runlog_vtk_keys = sorted(list(vtk_meshes_loaded.keys()))
             L.info(f"Final VTK meshes selected: {runlog_vtk_keys}")
        else:
            # This path is reached if VTK was needed but dir wasn't found/generated
            L.warning("No valid persistent 5ttgen directory found or generated for VTK loading.")
            vtk_meshes_loaded = {} # Ensure it's empty
    else:
        # VTK loading was not requested
        vtk_meshes_loaded = {}

    # --- Initial check if any surfaces are requested AT ALL ---
    if not exact_mesh_keys_requested and not base_cortical_needed and not cbm_bs_cc_needed:
        if vtk_load_needed and not vtk_meshes_loaded:
             L.error("VTK structures were requested, but none could be loaded. Check previous errors.")
        else:
             L.error("No surfaces specified or loaded. Please check arguments (e.g., --cortical-surfaces, --subcortical-gray).")
        sys.exit(0) # Exit gracefully if nothing was ever requested/loaded

    # --- External-tool sanity checks ---
    required_tools = []
    if cbm_bs_cc_needed: required_tools.append("mri_binarize") # For ASEG extraction
    if vtk_load_needed and persistent_5ttgen_path is None: # If VTK needed but failed generation
         required_tools.append("5ttgen") # Might still need 5ttgen check if generation failed
    if args.space.upper() == "MNI": required_tools.extend(["antsApplyTransforms", "warpinit", "mrcat"]) # For warping
    if required_tools:
        try:
            require_cmds(list(set(required_tools)), logger=L)
        except SystemExit:
            L.critical("Required external tools or libraries might be missing.")
            sys.exit(1)


    # --- Prepare Run Log (populate with determined values) ---
    runlog = {
        "tool": "brain_for_printing_surfaces", "mode": args.mode, "subject_id": args.subject_id, "space": args.space,
        "req_cortical": cortical_request, "req_cbm_bs_cc": cbm_bs_cc_request, "req_sgm": sgm_request, "req_ventricles": ventricle_request,
        "parsed_cortical_types": sorted(list(base_cortical_needed)), "parsed_cbm_bs_cc": sorted(list(cbm_bs_cc_needed)),
        "vtk_load_attempted": vtk_load_needed,
        # --- MODIFIED: Log the persistent path used ---
        "5ttgen_persistent_work_dir_used": str(persistent_5ttgen_path) if persistent_5ttgen_path else None,
        "loaded_vtk_keys": runlog_vtk_keys, # These are the filtered keys
        "final_mesh_keys_requested": sorted(list(exact_mesh_keys_requested)), # All keys from Cortical/ASEG + VTK
        "no_fill_structures": no_fill, "no_smooth_structures": no_smooth,
        "split_outputs": args.split_outputs, "output_dir": str(out_dir),
        "steps": [], "warnings": [], "output_files": [],
    }
    # Removed runlog_generation_warning as it's less relevant now
    if args.mode == "preset": runlog["preset_name"] = args.name

    # --- Main Processing ---
    generated_meshes: Dict[str, Optional[trimesh.Trimesh]] = {} # For Cortical/ASEG
    # Use a temporary directory specifically for ASEG/Cortical surface generation steps
    # This is separate from the persistent work_dir used for 5ttgen
    with temp_dir("surfgen_temp", keep=args.no_clean, base_dir=out_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        L.info(f"Using temporary folder for surface generation steps: {tmp_dir}")
        runlog["steps"].append(f"Created surface generation temp dir => {tmp_dir}")

        # --- Generate ASEG/Cortical/CBM-BS-CC surfaces ---
        # Only run if these types were requested
        if base_cortical_needed or cbm_bs_cc_needed:
             L.info(f"Requesting cortical types for generation: {base_cortical_needed}")
             L.info(f"Requesting CBM/BS/CC structures for generation: {cbm_bs_cc_needed}")
             try:
                 # Pass args.subjects_dir (which should be .../derivatives/)
                 # The generate_brain_surfaces function uses tmp_dir for its intermediate files
                 generated_meshes = generate_brain_surfaces(
                     subjects_dir=str(derivatives_dir_path), # Pass the resolved derivatives path
                     subject_id=args.subject_id, # Pass original ID with 'sub-' prefix
                     space=args.space,
                     surfaces=tuple(base_cortical_needed),
                     extract_structures=list(cbm_bs_cc_needed),
                     no_fill_structures=no_fill,
                     no_smooth_structures=no_smooth,
                     out_warp=args.out_warp, run=args.run, session=args.session,
                     verbose=args.verbose, tmp_dir=str(tmp_dir), # Pass the surfgen temp dir
                 )
                 runlog["steps"].append(f"ASEG/Cortical/CBM-BS-CC mesh generation process completed.")
             except ValueError as e: # Handles specific errors like inflated+MNI
                 L.error(f"Mesh generation failed: {e}")
                 L.critical("Cannot proceed due to generation error.")
                 sys.exit(1)
             except FileNotFoundError as e:
                 L.error(f"Mesh generation failed - required input file not found: {e}")
                 L.critical("Cannot proceed due to missing input.")
                 sys.exit(1)
             except Exception as e:
                 L.error(f"Unexpected error during mesh generation: {e}")
                 import traceback
                 traceback.print_exc()
                 sys.exit(1)
        else:
             L.info("Skipping ASEG/Cortical/CBM-BS-CC surface generation (not requested).")
             generated_meshes = {} # Ensure it's an empty dict

        # --- Combine generated + loaded VTK meshes ---
        # Use the filtered vtk_meshes_loaded dictionary here
        all_meshes = {**generated_meshes, **vtk_meshes_loaded}

        # --- Filter based on final requested keys ---
        # exact_mesh_keys_requested contains keys from Cortical/ASEG parsing + selected VTK keys
        meshes_to_export: Dict[str, trimesh.Trimesh] = {}
        L.info(f"Filtering for final export based on keys: {exact_mesh_keys_requested}")
        missing_keys = []
        for key in exact_mesh_keys_requested:
            if key in all_meshes and all_meshes[key] is not None and not all_meshes[key].is_empty:
                meshes_to_export[key] = all_meshes[key]
            else:
                L.warning(f"Requested key '{key}' not available or empty in final mesh set.")
                missing_keys.append(key)

        if not meshes_to_export:
            L.error("No requested meshes available for export after filtering.")
            if missing_keys: L.error(f"Missing/empty keys: {missing_keys}")
            log_base=f"surfaces_{args.mode}_failed";
            if args.mode=='preset':log_base+=f"_{args.name}";
            try:write_log(runlog, out_dir, base_name=log_base+"_log");
            except Exception as e_log: L.error(f"Failed write log: {e_log}");
            sys.exit(1) # Exit with error status

        # --- Export ---
        if args.split_outputs:
             L.info(f"Exporting {len(meshes_to_export)} meshes separately...")
             subject_label_clean = args.subject_id.replace('sub-', '')
             for name, mesh in meshes_to_export.items():
                 # Construct filename based on name/type
                 fname_parts = [f"sub-{subject_label_clean}"]
                 if args.session: fname_parts.append(f"ses-{args.session}")
                 if args.run: fname_parts.append(f"run-{args.run}")
                 fname_parts.append(f"space-{args.space}")

                 # Determine descriptor based on origin/type
                 if name.startswith("subcortical-"):
                     struct_part = name.split('-', 1)[1]; fname_parts.append(f"desc-{struct_part}_model-vtk")
                 elif name.startswith("ventricle-") or name.startswith("vessel-"):
                     struct_part = name.split('-', 1)[1]; fname_parts.append(f"desc-{struct_part}_model-vtk")
                 elif name.endswith("_L") or name.endswith("_R"):
                     base, hemi = name.rsplit("_", 1); fname_parts.append(f"hemi-{hemi}"); fname_parts.append(f"desc-{base}_surf") # Use surf suffix
                 elif name in CBM_BS_CC_CHOICES:
                     fname_parts.append(f"desc-{name}_model-aseg") # Use model suffix
                 else:
                     fname_parts.append(f"desc-{name}_model-unknown")

                 # Use stl suffix for output
                 fname = "_".join(fname_parts) + ".stl"; out_path = out_dir / fname
                 try:
                     L.info(f"Exporting {name} to {out_path}...");
                     if mesh and not mesh.is_empty:
                          mesh.export(out_path, file_type="stl")
                          runlog["steps"].append(f"Exported {name} => {out_path}")
                          runlog["output_files"].append(str(out_path))
                     else:
                          L.warning(f"Mesh '{name}' is empty or invalid, skipping export.")
                          runlog["warnings"].append(f"Skipped export of empty/invalid mesh: {name}")

                 except Exception as e:
                     L.error(f"Failed export {name}: {e}"); runlog["warnings"].append(f"Export fail {name}: {e}")
        else: # Merge
            L.info(f"Merging {len(meshes_to_export)} meshes...")
            mesh_list = [m for m in meshes_to_export.values() if m is not None and not m.is_empty] # Filter out None/empty
            combined_mesh = None
            if not mesh_list:
                 L.error("No valid meshes available for merging.")
                 runlog["warnings"].append("Merge skipped: No valid meshes found.")
            else:
                 try:
                     combined_mesh = trimesh.util.concatenate(mesh_list)
                     L.info(f"Concatenated {len(mesh_list)} meshes.")
                     runlog["steps"].append(f"Concatenated {len(mesh_list)} meshes")
                 except Exception as e:
                     L.error(f"Concatenation failed: {e}")
                     runlog["warnings"].append(f"Concatenation fail: {e}")

                 if combined_mesh and not combined_mesh.is_empty:
                      subject_label_clean = args.subject_id.replace('sub-', '')
                      fname_parts = [f"sub-{subject_label_clean}"]
                      if args.session: fname_parts.append(f"ses-{args.session}")
                      if args.run: fname_parts.append(f"run-{args.run}")
                      fname_parts.append(f"space-{args.space}")
                      desc = f"desc-combined-{args.mode}"
                      if args.mode == 'preset': desc += f"{args.name}"
                      fname_parts.append(desc)

                      out_fname = "_".join(fname_parts) + ".stl"; out_path = out_dir / out_fname
                      try:
                          L.info(f"Exporting combined mesh to {out_path}...")
                          combined_mesh.export(out_path, file_type="stl")
                          runlog["steps"].append(f"Exported merged mesh => {out_path}")
                          runlog["output_files"].append(str(out_path))
                      except Exception as e:
                          L.error(f"Export fail combined: {e}"); runlog["warnings"].append(f"Export fail combined: {e}")
                 else:
                     L.warning("Combined mesh is empty or invalid after concatenation, skipping export.")
                     runlog["warnings"].append("Combined mesh export skipped (empty/invalid).")

        # --- Cleanup or Temp Dir Warning ---
        # This refers to the surfgen_temp directory
        if args.no_clean:
            runlog["warnings"].append(f"Surface generation temp folder kept: {tmp_dir}")
            L.info(f"Surface generation temp folder kept: {tmp_dir}")

    # --- Save Log ---
    log_base = f"surfaces_{args.mode}"
    if args.mode == 'preset': log_base += f"_{args.name}"
    try:
        write_log(runlog, out_dir, base_name=log_base + "_log")
    except Exception as e:
        L.error(f"Failed write final JSON log: {e}")

    L.info("Script finished.")

# --- End of main() function ---

if __name__ == "__main__":
    log_level_main = logging.INFO
    logging.basicConfig(level=log_level_main, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)
    main()

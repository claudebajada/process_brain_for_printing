#!/usr/bin/env python
# brain_for_printing/cli_cortical_surfaces.py
#
# CLI for generating cortical surfaces and extracting other structures.
# Supports custom surface lists and predefined presets via subcommands.

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import trimesh
from typing import Dict, Optional, List, Set, Tuple # Import necessary types

# Assuming local imports are correct relative to this script's location
from .io_utils import temp_dir, require_cmds
from .log_utils import get_logger, write_log
from .surfaces import generate_brain_surfaces # Import the updated function
from . import constants as const # Import constants for structure name validation

# --- Constants for Argument Choices ---

# Base cortical types (implies both hemispheres unless specified)
CORTICAL_TYPES = ["pial", "white", "mid"]
# Hemisphere-specific cortical types
HEMI_CORTICAL_TYPES = [f"{h}-{t}" for h in ["lh", "rh"] for t in CORTICAL_TYPES]
# Extracted structure names (keys from STRUCTURE_LABEL_MAP in surfaces.py)
# Ensure these match the keys used in generate_brain_surfaces STRUCTURE_LABEL_MAP
STRUCTURE_NAMES = [
    "brainstem", "cerebellum_wm", "cerebellum_cortex",
    "cerebellum", "corpus_callosum"
]
# All possible choices for the --surfaces argument in custom mode
ALL_SURFACE_CHOICES = CORTICAL_TYPES + HEMI_CORTICAL_TYPES + STRUCTURE_NAMES

# Definitions for presets
PRESETS = {
    "pial_brain": ['lh-pial', 'rh-pial', 'corpus_callosum', 'cerebellum', 'brainstem'],
    "white_brain": ['lh-white', 'rh-white', 'corpus_callosum', 'cerebellum_wm', 'brainstem'],
    "mid_brain": ['lh-mid', 'rh-mid', 'corpus_callosum', 'cerebellum', 'brainstem'],
    "cortical_pial": ['lh-pial', 'corpus_callosum', 'rh-pial'],
    "cortical_white": ['lh-white', 'corpus_callosum', 'rh-white'],
    "cortical_mid": ['lh-mid', 'corpus_callosum', 'rh-mid'],
}

# Structures that might have fill/smooth options applied
FILL_SMOOTH_CHOICES = STRUCTURE_NAMES # Currently only non-cortical structures have fill/smooth

# Define logger at module level (will be configured in main)
L = logging.getLogger("brain_for_printing_surfaces") # Use a specific name

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #

def parse_surfaces_arg(requested_surfaces: List[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Parses the list from the --surfaces argument into:
    1. Base cortical types needed for generate_brain_surfaces (pial, white, mid)
    2. Extracted structure names for generate_brain_surfaces
    3. Exact mesh keys requested by the user for final export/combination
       (e.g., 'lh-pial' -> 'pial_L', 'pial' -> {'pial_L', 'pial_R'}, 'brainstem' -> 'brainstem')
    """
    base_cortical_needed: Set[str] = set()
    structures_to_extract: Set[str] = set()
    exact_mesh_keys_requested: Set[str] = set()

    for req_surf in requested_surfaces:
        if req_surf in CORTICAL_TYPES: # e.g., "pial"
            base_cortical_needed.add(req_surf)
            exact_mesh_keys_requested.add(f"{req_surf}_L")
            exact_mesh_keys_requested.add(f"{req_surf}_R")
        elif req_surf in HEMI_CORTICAL_TYPES: # e.g., "lh-pial"
            try:
                hemi_prefix, base_type = req_surf.split('-', 1) # lh, pial
                if hemi_prefix in ['lh', 'rh'] and base_type in CORTICAL_TYPES:
                    base_cortical_needed.add(base_type)
                    suffix = "_L" if hemi_prefix == 'lh' else "_R" # Correct key suffix
                    exact_mesh_keys_requested.add(f"{base_type}{suffix}")
                else:
                     L.warning(f"Ignoring malformed hemisphere surface request: {req_surf}")
            except ValueError:
                 L.warning(f"Ignoring malformed hemisphere surface request: {req_surf}")
        elif req_surf in STRUCTURE_NAMES: # e.g., "brainstem"
            structures_to_extract.add(req_surf)
            exact_mesh_keys_requested.add(req_surf)
        else:
            L.warning(f"Ignoring unrecognized surface request during parsing: {req_surf}")

    return base_cortical_needed, structures_to_extract, exact_mesh_keys_requested


# --------------------------------------------------------------------------- #
# CLI Argument Parser Setup
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate brain surfaces (cortical hemispheres, subcortical structures) "
            "and export as STL files. Use 'custom' mode to specify surfaces manually "
            "or 'preset' mode for predefined combinations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Create a parent parser for common arguments ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--subjects_dir", required=True, help="BIDS derivatives root.")
    parent_parser.add_argument("--subject_id", required=True, help="e.g. sub-01")
    parent_parser.add_argument("--output_dir", default=".", help="Directory to save output STL files.")
    parent_parser.add_argument("--space", choices=["T1", "MNI"], default="T1", help="Output space.")
    parent_parser.add_argument("--run", default=None, help="BIDS run entity.")
    parent_parser.add_argument("--session", default=None, help="BIDS session entity.")
    parent_parser.add_argument(
        "--split_outputs", action="store_true",
        help="Export each generated surface/structure separately instead of a single merged STL."
    )
    parent_parser.add_argument("--out_warp", default="warp.nii", help="Filename for 4-D warp if using MNI space.")
    parent_parser.add_argument("--no_clean", action="store_true", help="Keep temporary working folder.")
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")

    # --- Subparsers ---
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       title="Modes", help="Choose a mode: 'custom' or 'preset'.")

    # --- Custom Mode ---
    parser_custom = subparsers.add_parser(
        "custom",
        help="Specify surfaces/structures manually.",
        parents=[parent_parser] # Inherit common args
    )
    parser_custom.add_argument(
        "--surfaces",
        nargs='+',
        required=True,
        choices=ALL_SURFACE_CHOICES,
        metavar='SURFACE_OR_STRUCTURE',
        help=(f"List of surfaces/structures to generate (space-separated). "
              f"Cortical options (imply both hemispheres): {', '.join(CORTICAL_TYPES)}. "
              f"Hemisphere-specific: {', '.join(HEMI_CORTICAL_TYPES)}. "
              f"Other structures: {', '.join(STRUCTURE_NAMES)}.")
    )
    parser_custom.add_argument(
        "--no_fill_structures",
        nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE',
        help="List of extracted structures for which hole-filling should be skipped."
    )
    parser_custom.add_argument(
        "--no_smooth_structures",
        nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE',
        help="List of extracted structures for which smoothing should be skipped."
    )

    # --- Preset Mode ---
    parser_preset = subparsers.add_parser(
        "preset",
        help="Use predefined combinations of surfaces/structures.",
        parents=[parent_parser] # Inherit common args
    )
    parser_preset.add_argument(
        "--name",
        required=True,
        choices=list(PRESETS.keys()),
        help="Name of the preset combination to generate."
    )
    parser_preset.add_argument(
        "--no_fill_structures",
        nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE',
        help="List of structures within the preset to skip hole-filling on."
    )
    parser_preset.add_argument(
        "--no_smooth_structures",
        nargs='*', choices=FILL_SMOOTH_CHOICES, default=[], metavar='STRUCTURE',
        help="List of structures within the preset to skip smoothing on."
    )

    return parser


# --------------------------------------------------------------------------- #
# Main Execution Logic
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _build_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    # Configure the logger (use basicConfig for simplicity here)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)
    # Make sure the logger instance level is also set if basicConfig was called before
    L.setLevel(log_level)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine the list of surfaces to request ---
    if args.mode == "custom":
        requested_surfaces = args.surfaces
        no_fill = args.no_fill_structures
        no_smooth = args.no_smooth_structures
    elif args.mode == "preset":
        if args.name not in PRESETS:
             L.critical(f"Preset '{args.name}' is not defined in PRESETS dictionary.")
             sys.exit(1)
        requested_surfaces = PRESETS[args.name]
        no_fill = args.no_fill_structures
        no_smooth = args.no_smooth_structures
        L.info(f"Using preset '{args.name}': {', '.join(requested_surfaces)}")
    else:
        L.critical(f"Invalid mode '{args.mode}' encountered (should be caught by argparse).")
        sys.exit(1)

    # Parse the requested list
    base_cortical_needed, structures_to_extract, exact_mesh_keys_requested = parse_surfaces_arg(requested_surfaces)

    if not base_cortical_needed and not structures_to_extract:
        L.error("Parsing the requested surfaces yielded no valid items to generate.")
        sys.exit(1)

    # --- External-tool sanity checks ---
    required_tools = ["mri_binarize"]
    if args.space.upper() == "MNI":
        required_tools.extend(["antsApplyTransforms", "warpinit", "mrcat"])
    try:
        require_cmds(required_tools, logger=L)
    except SystemExit:
        L.critical("Required external tools not found. Please install them and ensure they are in your PATH.")
        sys.exit(1)

    # --- Prepare Run Log ---
    runlog = {
        "tool": "brain_for_printing_surfaces",
        "mode": args.mode,
        "subject_id": args.subject_id,
        "space": args.space,
        "requested_surfaces": requested_surfaces,
        "parsed_cortical_types": sorted(list(base_cortical_needed)),
        "parsed_extract_structures": sorted(list(structures_to_extract)),
        "no_fill_structures": no_fill,
        "no_smooth_structures": no_smooth,
        "split_outputs": args.split_outputs,
        "output_dir": str(out_dir),
        "steps": [],
        "warnings": [],
        "output_files": [],
    }
    if args.mode == "preset":
        runlog["preset_name"] = args.name

    # --- Work inside a managed temporary directory ---
    with temp_dir("surfaces", keep=args.no_clean, base_dir=out_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        L.info(f"Temporary folder: {tmp_dir}")
        runlog["steps"].append(f"Created temp dir => {tmp_dir}")

        # --- Generate surfaces and structures ---
        L.info(f"Requesting base cortical types from surfaces.py: {base_cortical_needed}")
        L.info(f"Requesting structure extraction from surfaces.py: {structures_to_extract}")

        all_meshes: Dict[str, Optional[trimesh.Trimesh]] = generate_brain_surfaces(
            subjects_dir=args.subjects_dir,
            subject_id=args.subject_id,
            space=args.space,
            surfaces=tuple(base_cortical_needed),
            extract_structures=list(structures_to_extract),
            no_fill_structures=no_fill,
            no_smooth_structures=no_smooth,
            out_warp=args.out_warp,
            run=args.run,
            session=args.session,
            verbose=args.verbose,
            tmp_dir=str(tmp_dir),
        )

        if not any(mesh is not None for mesh in all_meshes.values()):
             L.error("No meshes were generated successfully by generate_brain_surfaces. Exiting.")
             sys.exit(1)

        runlog["steps"].append(f"Core mesh generation completed in {args.space} space.")

        # --- Filter generated meshes based on exact user request ---
        meshes_to_export: Dict[str, trimesh.Trimesh] = {}
        L.info(f"Filtering generated meshes based on exact request: {exact_mesh_keys_requested}")
        for key in exact_mesh_keys_requested:
            if key in all_meshes and all_meshes[key] is not None:
                meshes_to_export[key] = all_meshes[key]
                L.debug(f"Keeping mesh '{key}' for export.")
            else:
                # This warning is expected if generate_brain_surfaces failed for a specific item
                L.warning(f"Requested mesh key '{key}' was not found in generated meshes or was None.")
                runlog["warnings"].append(f"Requested mesh '{key}' unavailable for export.")

        if not meshes_to_export:
            L.error("No requested meshes are available for export after filtering. Exiting.")
            sys.exit(1)

        # --- Export ---
        subject_label = args.subject_id.replace('sub-', '') # Clean subject ID once

        if args.split_outputs:
            L.info(f"Exporting {len(meshes_to_export)} requested meshes separately...")
            for name, mesh in meshes_to_export.items():
                # Construct BIDS-like filename
                fname_parts = [f"sub-{subject_label}"]
                if args.session: fname_parts.append(f"ses-{args.session}")
                if args.run: fname_parts.append(f"run-{args.run}")
                fname_parts.append(f"space-{args.space}")

                if name.endswith("_L") or name.endswith("_R"):
                     base, hemi = name.rsplit("_", 1)
                     fname_parts.append(f"label-{base}")
                     fname_parts.append(f"hemi-{hemi}")
                     fname_parts.append("surf")
                else:
                     fname_parts.append(f"desc-{name}")
                     fname_parts.append("surf")

                fname = "_".join(fname_parts) + ".stl"
                out_path = out_dir / fname
                try:
                    L.info(f"Exporting {name} to {out_path}...")
                    mesh.export(out_path, file_type="stl")
                    runlog["steps"].append(f"Exported {name} => {out_path}")
                    runlog["output_files"].append(str(out_path))
                except Exception as e:
                    L.error(f"Failed to export {name} to {out_path}: {e}")
                    runlog["warnings"].append(f"Export failed for {name}: {e}")

        else: # Merge and export
            L.info(f"Merging {len(meshes_to_export)} requested meshes...")
            mesh_list = list(meshes_to_export.values())
            combined_mesh = None # <--- Initialize before try block
            try:
                valid_mesh_list = [m for m in mesh_list if isinstance(m, trimesh.Trimesh)]
                if len(valid_mesh_list) != len(mesh_list):
                    L.warning("Some generated items were not valid meshes for concatenation.")
                if not valid_mesh_list:
                     raise ValueError("No valid meshes found to concatenate.")

                combined_mesh = trimesh.util.concatenate(valid_mesh_list)
                L.info(f"Concatenated {len(valid_mesh_list)} meshes.")
            except Exception as e:
                L.error(f"Failed to concatenate meshes: {e}")
                runlog["warnings"].append(f"Mesh concatenation failed: {e}")

            # Now check the combined_mesh variable, which is guaranteed to exist
            if combined_mesh and not combined_mesh.is_empty:
                 # Create a descriptive filename
                 fname_parts = [f"sub-{subject_label}"]
                 if args.session: fname_parts.append(f"ses-{args.session}")
                 if args.run: fname_parts.append(f"run-{args.run}")
                 fname_parts.append(f"space-{args.space}")

                 if args.mode == 'preset':
                     desc = f"preset-{args.name}"
                 else:
                     desc = "desc-custom"
                 fname_parts.append(desc)
                 fname_parts.append("combined")

                 out_fname = "_".join(fname_parts) + ".stl"
                 out_path = out_dir / out_fname
                 try:
                     L.info(f"Exporting combined mesh to {out_path}...")
                     combined_mesh.export(out_path, file_type="stl")
                     runlog["steps"].append(f"Exported merged mesh => {out_path}")
                     runlog["output_files"].append(str(out_path))
                 except Exception as e:
                     L.error(f"Failed to export combined mesh to {out_path}: {e}")
                     runlog["warnings"].append(f"Export failed for combined mesh: {e}")
            else:
                 L.warning("Combined mesh is empty or invalid after concatenation attempt, skipping export.")


        if args.no_clean:
            runlog["warnings"].append(f"Temporary folder kept via --no_clean: {tmp_dir}")
        # temp_dir context manager handles cleanup otherwise

    # --- Save JSON audit-log ---
    log_base = f"surfaces_{args.mode}"
    if args.mode == 'preset':
        log_base += f"_{args.name}"
    try:
        write_log(runlog, out_dir, base_name=log_base + "_log")
    except Exception as e:
        L.error(f"Failed to write JSON run log: {e}")

    L.info("Script finished.")


if __name__ == "__main__":
    # Set up basic logging configuration here if the script is run directly
    log_level_main = logging.INFO # Default level
    logging.basicConfig(level=log_level_main, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)
    main()

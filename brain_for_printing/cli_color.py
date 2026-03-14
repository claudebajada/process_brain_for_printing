#!/usr/bin/env python
# brain_for_printing/cli_color.py
#
# Two-mode command:
#   1) direct  – color an existing mesh from a param map.
#   2. preset  – generate a preset group of surfaces, color only the
#                cortical and cerebellar components, and combine all.
#
# Uses shared helpers and configuration utilities.

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
import trimesh
import os
from typing import List, Dict, Set, Tuple, Optional

# --- Local Imports ---
from .io_utils import temp_dir, flexible_match
from .log_utils import get_logger, write_log
from .color_utils import project_param_to_surface, copy_vertex_colors
from .mesh_utils import gifti_to_trimesh
from .warp_utils import create_mrtrix_warp, warp_gifti_vertices
# Import surface generation function
from .surfaces import generate_brain_surfaces
# Import shared constants and preset utilities
from . import constants as const
from .config_utils import PRESETS, parse_preset
# --- End Imports ---

# Map surface type arguments to BIDS suffixes
SURF_ARG_TO_BIDS_SUFFIX = {
    "pial": "pial", "mid": "midthickness", "white": "smoothwm", "inflated": "inflated"
}

# Define cerebellum keys explicitly for checking which components to color
CEREBELLUM_KEYS = {'cerebellum', 'cerebellum_wm', 'cerebellum_cortex'}

# Define valid choices for the color sampling surface (excluding 'inflated')
VALID_SAMPLING_SURFACES = [t for t in const.CORTICAL_TYPES if t != 'inflated']

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Color brain meshes using a parameter map. Can color an existing mesh "
            "directly or generate surfaces based on a preset before coloring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True, title="Subcommands")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--param_map", required=True, help="Path to NIfTI parameter map for coloring.")
    common_parser.add_argument("--param_threshold", type=float, default=None, help="Vertices below this value get different alpha/color (optional).")
    common_parser.add_argument("--num_colors", type=int, default=6, help="Number of discrete colors for the map.")
    common_parser.add_argument("--order", type=int, default=1, help="Interpolation order for sampling param map (0=nearest, 1=linear).")
    common_parser.add_argument("--output_dir", default=".", help="Output directory.")
    common_parser.add_argument("--no_clean", action="store_true", help="Keep temporary files/folders.")
    common_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging (shows DEBUG messages).")

    d = sub.add_parser("direct", help="Color an existing mesh file directly.", parents=[common_parser])
    d.add_argument("--in_mesh", required=True, help="Path to the input mesh file (STL, OBJ, GIFTI, etc.).")
    d.add_argument("--out_obj", required=False, default=None, help="Explicit name for the output colored mesh file (OBJ format recommended). If not set, defaults to a generated name in --output_dir.")

    preset_parser = sub.add_parser("preset", help="Generate surfaces based on a preset, then color the cortical/cerebellar components.", parents=[common_parser])
    preset_parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory (e.g., /path/to/bids/derivatives).")
    preset_parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-CB).")
    preset_parser.add_argument("--preset", required=True, choices=list(PRESETS.keys()), help="Name of the surface generation preset to use.")
    preset_parser.add_argument("--space",
                             default="T1",
                             help="Space for surface generation: 'T1' (native), 'MNI' (template), or target subject ID (e.g., 'sub-01') to warp into that subject's T1 space.")
    preset_parser.add_argument("--color_in",
                             required=True,
                             choices=["source", "target"],
                             help="Space in which to perform coloring. 'source' colors in the original subject's T1 space before warping, 'target' colors in the final space (T1/MNI/target subject). The parameter map must be in the specified space.")
    preset_parser.add_argument(
        "--color_sampling_surf",
        type=str,
        default=None,
        choices=VALID_SAMPLING_SURFACES, 
        help=(
            "Optional: Specify a cortical surface type to use for sampling the "
            "color map for cortical targets. Choices: {}. If set, colors are sampled "
            "using this surface's geometry and transferred to the target preset "
            "surfaces. Requires vertex correspondence. Default is to sample using "
            "the target surface itself."
        ).format(VALID_SAMPLING_SURFACES)
    )
    preset_parser.add_argument("--run", default=None, help="BIDS run entity (optional).")
    preset_parser.add_argument("--session", default=None, help="BIDS session entity (optional).")

    return parser

# --------------------------------------------------------------------------- #
# Mode implementations
# --------------------------------------------------------------------------- #
def _run_direct(args, logger) -> None:
    """Executes the 'direct' coloring subcommand."""
    if not Path(args.in_mesh).is_file(): logger.error(f"Input mesh not found: {args.in_mesh}"); sys.exit(1)
    if not Path(args.param_map).is_file(): logger.error(f"Parameter map not found: {args.param_map}"); sys.exit(1)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_obj:
        out_path = Path(args.out_obj)
        if out_path.parent == Path('.'): out_path = out_dir / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        in_stem = Path(args.in_mesh).stem
        for suffix in ['.surf', '.gii', '.obj', '.stl']: # More comprehensive suffix removal
             if in_stem.endswith(suffix): in_stem = in_stem[:-len(suffix)]; break
        map_stem = Path(args.param_map).stem.replace('.nii','').replace('.gz','')
        out_fname = f"{in_stem}_map-{map_stem}_colored.obj"; out_path = out_dir / out_fname
    
    logger.info(f"Output will be saved to: {out_path}")
    runlog = {"tool": "brain_for_printing_color_direct", "in_mesh": args.in_mesh, "param_map": args.param_map, "param_threshold": args.param_threshold, "out_obj": str(out_path), "num_colors": args.num_colors, "order": args.order, "steps": [], "warnings": [], "output_files": []}
    
    try:
        with temp_dir("color_direct", keep=args.no_clean, base_dir=str(out_dir)) as tmp_str: # Ensure base_dir is str
            tmp = Path(tmp_str) # Work with Path object
            logger.info(f"Temporary folder (if needed for intermediate steps): {tmp}")
            
            logger.info(f"Loading mesh: {args.in_mesh}")
            if args.in_mesh.lower().endswith(".gii"): # Case-insensitive check
                mesh = gifti_to_trimesh(args.in_mesh)
            else: 
                mesh = trimesh.load(args.in_mesh, force='mesh')
            runlog["steps"].append("Loaded mesh");
            
            if mesh.is_empty: 
                logger.error("Input mesh is empty.")
                raise ValueError("Input mesh is empty.")

            logger.info(f"Applying parameter map: {args.param_map}")
            project_param_to_surface(mesh=mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)
            runlog["steps"].append("Applied param-map colouring")
            
            logger.info(f"Exporting colored mesh to {out_path} (OBJ format)")
            mesh.export(out_path, file_type="obj")
            runlog["steps"].append(f"Exported OBJ => {out_path}"); runlog["output_files"].append(str(out_path))
            
    except Exception as e:
        logger.error(f"Error during direct coloring: {e}", exc_info=args.verbose)
        runlog["warnings"].append(f"Execution failed: {e}")
        write_log(runlog, str(out_dir), base_name="color_direct_failed_log"); sys.exit(1) # Ensure out_dir is str
        
    write_log(runlog, str(out_dir), base_name="color_direct_log") # Ensure out_dir is str
    logger.info("Direct coloring finished successfully.")


def _run_preset(args, logger) -> None:
    """Executes the 'preset' coloring subcommand."""
    if not Path(args.param_map).is_file(): logger.error(f"Parameter map not found: {args.param_map}"); sys.exit(1)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    runlog = { 
        "tool": "brain_for_printing_color_preset", "preset": args.preset, 
        "subject_id": args.subject_id, "space": args.space, 
        "param_map": args.param_map, "param_threshold": args.param_threshold, 
        "color_sampling_surf": args.color_sampling_surf, 
        "num_colors": args.num_colors, "order": args.order, 
        "output_dir": str(out_dir), "color_in": args.color_in, 
        "steps": [], "warnings": [], "output_files": [] 
    }

    try:
        base_cortical_needed, other_structures_needed, exact_mesh_keys = parse_preset(args.preset)
        logger.info(f"Preset '{args.preset}' requires cortical types: {base_cortical_needed or 'None'}")
        logger.info(f"Preset '{args.preset}' requires other structures: {other_structures_needed or 'None'}")
        logger.debug(f"Preset '{args.preset}' expects exact mesh keys: {exact_mesh_keys}")
        runlog["preset_cortical_types"] = sorted(list(base_cortical_needed))
        runlog["preset_other_structures"] = sorted(list(other_structures_needed))
        runlog["preset_expected_keys"] = exact_mesh_keys

        if args.space.upper() == "MNI" and "inflated" in base_cortical_needed:
             logger.error("Inflated surfaces cannot be generated or colored in MNI space.")
             raise ValueError("Inflated surfaces cannot be generated or colored in MNI space.")

        with temp_dir("color_preset_gen", keep=args.no_clean, base_dir=str(out_dir)) as tmp_str: # Ensure base_dir is str
            tmp = Path(tmp_str) # Work with Path object
            logger.info(f"Temporary folder for surface generation: {tmp}")
            runlog["steps"].append(f"Created temp dir => {tmp}")

            source_meshes_for_coloring = {}
            if args.color_in == "source":
                logger.info("Getting T1-native meshes for coloring in source space...")
                # Only fetch what's needed for coloring (cortical + cerebellar from preset)
                colorable_cortical = base_cortical_needed
                colorable_other = {s for s in other_structures_needed if s in CEREBELLUM_KEYS}

                source_meshes_for_coloring = generate_brain_surfaces(
                    subjects_dir=args.subjects_dir, subject_id=args.subject_id,
                    space="T1", surfaces=tuple(colorable_cortical),
                    extract_structures=list(colorable_other), run=args.run,
                    session=args.session, verbose=args.verbose, tmp_dir=str(tmp),
                )
                runlog["steps"].append("Generated/loaded source meshes needed for coloring")

            logger.info(f"Generating TARGET surfaces ({args.space} space) required by preset...")
            target_meshes = generate_brain_surfaces(
                subjects_dir=args.subjects_dir, subject_id=args.subject_id,
                space=args.space, surfaces=tuple(base_cortical_needed),
                extract_structures=list(other_structures_needed), run=args.run,
                session=args.session, verbose=args.verbose, tmp_dir=str(tmp),
            )
            runlog["steps"].append("Target surface generation process completed")

            final_meshes_to_combine = []
            processed_keys_count = 0
            colored_keys_list = []
            logger.info("Processing generated surfaces (coloring cortical/cerebellar components)...")
            subject_label_clean = args.subject_id.replace('sub-', '')
            
            # Determine anat_dir for fetching sampling surfaces if needed
            anat_dir_path_str = str(Path(args.subjects_dir) / f"sub-{subject_label_clean}" / "anat")


            for key in exact_mesh_keys:
                target_mesh_component = target_meshes.get(key)
                if target_mesh_component is None or target_mesh_component.is_empty:
                    logger.warning(f"Target mesh component '{key}' not found or is empty. Skipping.")
                    runlog["warnings"].append(f"Skipped missing/empty target component: {key}")
                    continue

                # Determine if this component should be colored
                component_type_for_coloring = None; component_hemi_for_coloring = None
                is_cortical_component = False; is_cerebellum_component = False

                if key.endswith("_L"): component_type_for_coloring = key[:-2]; component_hemi_for_coloring = "L"
                elif key.endswith("_R"): component_type_for_coloring = key[:-2]; component_hemi_for_coloring = "R"
                elif key in const.CBM_BS_CC_CHOICES: component_type_for_coloring = key
                
                if component_hemi_for_coloring and component_type_for_coloring in const.CORTICAL_TYPES: is_cortical_component = True
                if component_type_for_coloring in CEREBELLUM_KEYS: is_cerebellum_component = True
                
                should_color_this_component = is_cortical_component or is_cerebellum_component

                if should_color_this_component:
                    logger.debug(f"Attempting to color component: {key}")
                    
                    use_alternative_sampling_surface = False
                    sampling_surface_type_from_arg = args.color_sampling_surf
                    # Check if alternative sampling is requested and applicable
                    if is_cortical_component and sampling_surface_type_from_arg and \
                       sampling_surface_type_from_arg != component_type_for_coloring:
                        use_alternative_sampling_surface = True
                        logger.info(f"Coloring target cortical component '{key}' by sampling from specified '{sampling_surface_type_from_arg}' surface type.")
                    elif is_cortical_component: logger.info(f"Coloring cortical target '{key}' by sampling from itself.")
                    elif is_cerebellum_component: logger.info(f"Coloring cerebellar target '{key}' by sampling from itself.")

                    try:
                        mesh_to_color_or_copy_from = None # This will hold the mesh that gets colors applied

                        if args.color_in == "source":
                            mesh_to_color_or_copy_from = source_meshes_for_coloring.get(key)
                            if mesh_to_color_or_copy_from is None or mesh_to_color_or_copy_from.is_empty:
                                raise ValueError(f"Required source mesh for coloring component {key} not found or empty.")
                        else: # color_in == "target"
                            mesh_to_color_or_copy_from = target_mesh_component # Color the target mesh directly

                        # --- Actual Coloring or Color Transfer Logic ---
                        if use_alternative_sampling_surface:
                            # Find the BIDS suffix for the alternative sampling surface type
                            alternative_sampling_bids_suffix = SURF_ARG_TO_BIDS_SUFFIX.get(sampling_surface_type_from_arg)
                            if not alternative_sampling_bids_suffix:
                                raise ValueError(f"Invalid --color_sampling_surf type '{sampling_surface_type_from_arg}' resulted in no BIDS suffix.")
                            
                            logger.debug(f"Alternative sampling: using surf type '{sampling_surface_type_from_arg}' (suffix: {alternative_sampling_bids_suffix}) for hemi '{component_hemi_for_coloring}'.")

                            # Path to the alternative sampling GIFTI file (always T1 native initially)
                            # This assumes flexible_match can find it in the subject's anat dir
                            sampling_gii_native_t1_path_str = flexible_match(
                                base_dir=anat_dir_path_str, subject_id=args.subject_id, # Use full subject_id for flexible_match
                                suffix=f"{alternative_sampling_bids_suffix}.surf", 
                                hemi=component_hemi_for_coloring, # flexible_match handles hemi prefix
                                ext=".gii", run=args.run, session=args.session, logger=logger
                            )
                            sampling_gii_path_to_use = Path(sampling_gii_native_t1_path_str)
                            
                            if not sampling_gii_path_to_use.is_file():
                                raise FileNotFoundError(f"Alternative sampling surface file not found: {sampling_gii_path_to_use}")
                            runlog["steps"].append(f"Located alternative sampling surface {sampling_surface_type_from_arg} for {key}: {sampling_gii_path_to_use.name}")

                            # If coloring in target space AND target space is MNI, warp the T1 sampling surface to MNI
                            if args.color_in == "target" and args.space.upper() == "MNI":
                                logger.info(f"Warping T1 native sampling surface '{sampling_gii_path_to_use.name}' to MNI space for component {key}...")
                                # Find necessary files for warping T1 to MNI
                                t1_prep_for_warp = flexible_match( anat_dir_path_str, args.subject_id, descriptor="preproc", suffix="T1w", ext=".nii.gz", session=args.session, run=args.run, logger=logger)
                                mni_ref_for_warp = flexible_match( anat_dir_path_str, args.subject_id, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", session=args.session, run=args.run, logger=logger)
                                mni_to_t1_xfm_for_warp = flexible_match( anat_dir_path_str, args.subject_id, descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image", suffix="xfm", ext=".h5", session=args.session, run=args.run, logger=logger)
                                
                                # Define warp field path (specific to this sampling surface to avoid conflicts if multiple used)
                                t1_to_mni_warp_field = tmp / f"warp_sampling_{sampling_surface_type_from_arg}_{component_hemi_for_coloring}_T1w-to-MNI.nii.gz"
                                if not t1_to_mni_warp_field.exists(): # Create warp only if it doesn't exist yet
                                    create_mrtrix_warp(str(mni_ref_for_warp), str(t1_prep_for_warp), str(mni_to_t1_xfm_for_warp), str(t1_to_mni_warp_field), str(tmp), args.verbose)
                                
                                # Define warped sampling surface path
                                sampling_gii_warped_mni_path = tmp / f"sampling_{sampling_surface_type_from_arg}_{component_hemi_for_coloring}_space-MNI.surf.gii"
                                warp_gifti_vertices(str(sampling_gii_path_to_use), str(t1_to_mni_warp_field), str(sampling_gii_warped_mni_path), args.verbose)
                                
                                if not sampling_gii_warped_mni_path.exists():
                                    raise FileNotFoundError(f"Failed to create MNI-warped alternative sampling surface: {sampling_gii_warped_mni_path}")
                                sampling_gii_path_to_use = sampling_gii_warped_mni_path # Update to use the MNI version
                                logger.info(f"Using MNI-warped alternative sampling surface: {sampling_gii_path_to_use.name}")
                                runlog["steps"].append(f"Warped alt. sampling surf {sampling_gii_path_to_use.name} to MNI for {key}")
                            
                            # Load the (potentially warped) alternative sampling mesh
                            alternative_sampling_mesh = gifti_to_trimesh(str(sampling_gii_path_to_use))
                            if alternative_sampling_mesh.is_empty:
                                raise ValueError(f"Alternative sampling mesh {sampling_gii_path_to_use.name} is empty.")

                            logger.debug(f"Coloring the alternative sampling mesh ('{sampling_surface_type_from_arg}') itself...")
                            project_param_to_surface(mesh=alternative_sampling_mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)
                            
                            # Transfer colors to the actual mesh_to_color_or_copy_from
                            logger.debug(f"Copying colors from alt. sampling mesh to actual component mesh ('{key}')...")
                            if len(alternative_sampling_mesh.vertices) != len(mesh_to_color_or_copy_from.vertices):
                                raise ValueError(f"Vertex count mismatch: alt. sampling ({len(alternative_sampling_mesh.vertices)}) vs actual component ({len(mesh_to_color_or_copy_from.vertices)}) for key '{key}'.")
                            copy_vertex_colors(alternative_sampling_mesh, mesh_to_color_or_copy_from)

                        else: # No alternative sampling surface, color the component mesh directly
                            logger.debug(f"Coloring component mesh '{key}' directly (sampling from itself)...")
                            project_param_to_surface(mesh=mesh_to_color_or_copy_from, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)

                        # If coloring was done in source space, now copy these colors to the final target_mesh_component
                        if args.color_in == "source":
                            logger.debug(f"Final step for {key}: Copying colors from (colored) source-space mesh to target-space mesh.")
                            if len(mesh_to_color_or_copy_from.vertices) != len(target_mesh_component.vertices): # mesh_to_color_or_copy_from is the source mesh here
                                raise ValueError(f"Vertex count mismatch: source ({len(mesh_to_color_or_copy_from.vertices)}) vs target ({len(target_mesh_component.vertices)}) for key '{key}' during final color copy.")
                            copy_vertex_colors(mesh_to_color_or_copy_from, target_mesh_component)
                        
                        # If coloring was in target (and no alt sampling, or alt sampling was on target directly), colors are already on target_mesh_component
                        # If coloring was in target with alt sampling, colors were put on alt_sampling_mesh and then copied to mesh_to_color_or_copy_from (which is target_mesh_component)

                        runlog["steps"].append(f"Applied coloring to component {key} (space: {args.color_in})")
                        colored_keys_list.append(key)

                    except Exception as e_color:
                        logger.warning(f"Failed to process/color component {key}: {e_color}", exc_info=args.verbose)
                        runlog["warnings"].append(f"Processing/Coloring failed for {key}: {str(e_color)[:100]}") # Log snippet of error
                        # Add the uncolored target_mesh_component to the list to combine anyway
                        final_meshes_to_combine.append(target_mesh_component)
                        processed_keys_count += 1
                        continue # Skip to next component
                else:
                    logger.info(f"Skipping coloring for non-cortical/non-cerebellar component: {key}")
                    runlog["steps"].append(f"Skipped coloring for {key} (not cortical/cerebellar)")

                final_meshes_to_combine.append(target_mesh_component) # Add the (potentially colored) target mesh
                processed_keys_count += 1

            if not final_meshes_to_combine: 
                logger.error("No mesh components successfully processed or generated to combine.")
                raise RuntimeError("No mesh components successfully generated/processed.")

            logger.info(f"Combining {len(final_meshes_to_combine)} mesh components...")
            combined_mesh = trimesh.util.concatenate(final_meshes_to_combine)
            runlog["steps"].append(f"Combined {len(final_meshes_to_combine)} components (colored: {', '.join(colored_keys_list) or 'None'}). Total processed: {processed_keys_count}")
            
            if combined_mesh.is_empty: 
                logger.error("Combined mesh is empty after concatenation.")
                raise RuntimeError("Combined mesh is empty after concatenation.")

            # Construct filename
            fname_parts = [args.subject_id] # Start with full subject ID
            if args.session: fname_parts.append(f"ses-{args.session.replace('ses-','')}")
            if args.run: fname_parts.append(f"run-{args.run.replace('run-','')}")
            fname_parts.append(f"space-{args.space}")
            fname_parts.append(f"preset-{args.preset}")
            map_stem = Path(args.param_map).stem.replace('.nii','').replace('.gz','')
            fname_parts.append(f"map-{map_stem}")
            if args.color_sampling_surf: fname_parts.append(f"sampling-{args.color_sampling_surf}")
            out_fname = "_".join(fname_parts) + "_selectively_colored.obj"
            out_path = out_dir / out_fname

            logger.info(f"Exporting combined selectively colored mesh to {out_path} (OBJ format)")
            combined_mesh.export(out_path, file_type="obj")
            runlog["output_files"].append(str(out_path))
            runlog["steps"].append(f"Exported combined selectively colored OBJ => {out_path}")

    except Exception as e:
        logger.error(f"Error during preset processing: {e}", exc_info=args.verbose)
        runlog["warnings"].append(f"Execution failed: {str(e)}")
        write_log(runlog, str(out_dir), base_name=f"color_preset_{args.preset}_failed_log") # Ensure out_dir is str
        sys.exit(1)

    write_log(runlog, str(out_dir), base_name=f"color_preset_{args.preset}_log") # Ensure out_dir is str
    logger.info("Preset processing finished successfully.")


# --------------------------------------------------------------------------- #
# Main dispatcher
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _build_parser().parse_args()
    
    # Configure logging: INFO by default, DEBUG if verbose
    log_level = logging.DEBUG if args.verbose else logging.INFO # MODIFIED LINE
    L = get_logger("brain_for_printing_color", level=log_level)

    if args.subcommand == "direct":
        _run_direct(args, L)
    elif args.subcommand == "preset":
        _run_preset(args, L)
    else:
        # This case should not be reached due to argparse `required=True` for subcommands
        L.critical(f"Unknown subcommand: {args.subcommand if hasattr(args, 'subcommand') else 'None'}")
        sys.exit(1)


if __name__ == "__main__":
    main()

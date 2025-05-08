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

L = logging.getLogger("brain_for_printing_color")

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
    common_parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")

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
    # --- MODIFIED: Restricted choices for --color_sampling_surf ---
    preset_parser.add_argument(
        "--color_sampling_surf",
        type=str,
        default=None,
        choices=VALID_SAMPLING_SURFACES, # Use the restricted list
        help=(
            "Optional: Specify a cortical surface type to use for sampling the "
            "color map for cortical targets. Choices: {}. If set, colors are sampled "
            "using this surface's geometry and transferred to the target preset "
            "surfaces. Requires vertex correspondence. Default is to sample using "
            "the target surface itself."
        ).format(VALID_SAMPLING_SURFACES)
    )
    # --- End Modification ---
    preset_parser.add_argument("--run", default=None, help="BIDS run entity (optional).")
    preset_parser.add_argument("--session", default=None, help="BIDS session entity (optional).")

    return parser

# --------------------------------------------------------------------------- #
# Mode implementations (No changes needed in the core logic below this point)
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
        for suffix in ['.surf', '.gii', '.obj', '.stl']:
             if in_stem.endswith(suffix): in_stem = in_stem[:-len(suffix)]; break
        map_stem = Path(args.param_map).stem.replace('.nii','').replace('.gz','')
        out_fname = f"{in_stem}_map-{map_stem}_colored.obj"; out_path = out_dir / out_fname
    logger.info(f"Output will be saved to: {out_path}")
    runlog = {"tool": "brain_for_printing_color_direct", "in_mesh": args.in_mesh, "param_map": args.param_map, "param_threshold": args.param_threshold, "out_obj": str(out_path), "num_colors": args.num_colors, "order": args.order, "steps": [], "warnings": [], "output_files": []}
    try:
        with temp_dir("color_direct", keep=args.no_clean, base_dir=out_dir) as tmp:
            logger.info(f"Temp folder (if needed): {tmp}")
            logger.info(f"Loading mesh: {args.in_mesh}")
            if args.in_mesh.endswith(".gii"): mesh = gifti_to_trimesh(args.in_mesh)
            else: mesh = trimesh.load(args.in_mesh, force='mesh')
            runlog["steps"].append("Loaded mesh");
            if mesh.is_empty: raise ValueError("Input mesh is empty.")
            logger.info(f"Applying parameter map: {args.param_map}")
            project_param_to_surface(mesh=mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)
            runlog["steps"].append("Applied param-map colouring")
            logger.info(f"Exporting colored mesh to {out_path} (OBJ format)")
            mesh.export(out_path, file_type="obj")
            runlog["steps"].append(f"Exported OBJ => {out_path}"); runlog["output_files"].append(str(out_path))
    except Exception as e:
        logger.error(f"Error during direct coloring: {e}", exc_info=args.verbose)
        runlog["warnings"].append(f"Execution failed: {e}")
        write_log(runlog, out_dir, base_name="color_direct_failed_log"); sys.exit(1)
    write_log(runlog, out_dir, base_name="color_direct_log")
    logger.info("Direct coloring finished successfully.")


def _run_preset(args, logger) -> None:
    """Executes the 'preset' coloring subcommand."""
    if not Path(args.param_map).is_file(): logger.error(f"Parameter map not found: {args.param_map}"); sys.exit(1)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runlog = { "tool": "brain_for_printing_color_preset", "preset": args.preset, "subject_id": args.subject_id, "space": args.space, "param_map": args.param_map, "param_threshold": args.param_threshold, "color_sampling_surf": args.color_sampling_surf, "num_colors": args.num_colors, "order": args.order, "output_dir": str(out_dir), "color_in": args.color_in, "steps": [], "warnings": [], "output_files": [] }

    try:
        base_cortical_needed, other_structures_needed, exact_mesh_keys = parse_preset(args.preset)
        logger.info(f"Preset '{args.preset}' requires cortical types: {base_cortical_needed}")
        logger.info(f"Preset '{args.preset}' requires other structures: {other_structures_needed}")
        logger.info(f"Preset '{args.preset}' expects exact mesh keys: {exact_mesh_keys}")
        runlog["preset_cortical_types"] = sorted(list(base_cortical_needed))
        runlog["preset_other_structures"] = sorted(list(other_structures_needed))
        runlog["preset_expected_keys"] = exact_mesh_keys

        # Check MNI incompatibility (generate_brain_surfaces also checks target generation)
        if args.space.upper() == "MNI" and "inflated" in base_cortical_needed:
             raise ValueError("Inflated surfaces cannot be generated or colored in MNI space.")
        # The parser already prevents args.color_sampling_surf from being 'inflated'

        with temp_dir("color_preset_gen", keep=args.no_clean, base_dir=out_dir) as tmp:
            logger.info(f"Temporary folder for surface generation: {tmp}")
            runlog["steps"].append(f"Created temp dir => {tmp}")

            # If coloring in source space, we need to get the source meshes first
            source_meshes = {}
            if args.color_in == "source":
                logger.info("Getting source meshes for coloring in source space...")
                source_meshes = generate_brain_surfaces(
                    subjects_dir=args.subjects_dir, subject_id=args.subject_id,
                    space="T1", surfaces=tuple(base_cortical_needed),
                    extract_structures=list(other_structures_needed), run=args.run,
                    session=args.session, verbose=args.verbose, tmp_dir=str(tmp),
                )
                runlog["steps"].append("Generated source meshes for coloring")

            logger.info("Generating TARGET surfaces required by preset...")
            target_meshes = generate_brain_surfaces(
                subjects_dir=args.subjects_dir, subject_id=args.subject_id,
                space=args.space, surfaces=tuple(base_cortical_needed),
                extract_structures=list(other_structures_needed), run=args.run,
                session=args.session, verbose=args.verbose, tmp_dir=str(tmp),
            )
            runlog["steps"].append("Target surface generation process completed")

            final_meshes_to_combine = []
            processed_keys = []
            colored_keys = []
            logger.info("Processing generated surfaces (coloring cortical/cerebellar)...")
            subject_label_clean = args.subject_id.replace('sub-', '')
            anat_dir = Path(args.subjects_dir) / f"sub-{subject_label_clean}" / "anat"

            for key in exact_mesh_keys:
                target_mesh = target_meshes.get(key)
                if target_mesh is None or target_mesh.is_empty:
                    logger.warning(f"Target mesh component '{key}' not found/empty. Skipping.")
                    runlog["warnings"].append(f"Skipped missing/empty target component: {key}")
                    continue

                target_type = None; target_hemi = None
                is_cortical = False; is_cerebellum = False
                if key.endswith("_L"): target_type = key[:-2]; target_hemi = "L"
                elif key.endswith("_R"): target_type = key[:-2]; target_hemi = "R"
                elif key in const.CBM_BS_CC_CHOICES: target_type = key
                if target_hemi and target_type in const.CORTICAL_TYPES: is_cortical = True
                if target_type in CEREBELLUM_KEYS: is_cerebellum = True

                should_color_component = is_cortical or is_cerebellum

                if should_color_component:
                    logger.debug(f"Coloring component: {key}")
                    use_cross_sampling = False
                    sampling_surf_type_arg = args.color_sampling_surf
                    if is_cortical and sampling_surf_type_arg is not None and sampling_surf_type_arg != target_type:
                        use_cross_sampling = True
                        logger.info(f"Coloring target '{key}' by sampling from '{sampling_surf_type_arg}' surface.")
                    elif is_cortical: logger.info(f"Coloring cortical target '{key}' by sampling from itself.")
                    elif is_cerebellum: logger.info(f"Coloring cerebellar target '{key}' by sampling from itself.")

                    try:
                        if args.color_in == "source":
                            # Color in source space and then warp to target
                            source_mesh = source_meshes.get(key)
                            if source_mesh is None or source_mesh.is_empty:
                                raise ValueError(f"Source mesh for {key} not found or empty")
                            
                            if use_cross_sampling:
                                sampling_bids_suffix = SURF_ARG_TO_BIDS_SUFFIX.get(sampling_surf_type_arg)
                                if not sampling_bids_suffix: raise ValueError(f"Invalid --color_sampling_surf '{sampling_surf_type_arg}'")
                                logger.debug(f"Finding sampling surface: hemi={target_hemi}, suffix={sampling_bids_suffix}")
                                sampling_gii_path_str = flexible_match(base_dir=anat_dir, subject_id=f"sub-{subject_label_clean}", suffix=f"{sampling_bids_suffix}.surf", hemi=f"hemi-{target_hemi}", ext=".gii", run=args.run, session=args.session)
                                sampling_gii_path = Path(sampling_gii_path_str)
                                if not sampling_gii_path.is_file(): raise FileNotFoundError(f"Sampling surface file not found: {sampling_gii_path}")
                                runlog["steps"].append(f"Located sampling surface {sampling_surf_type_arg} for {key}: {sampling_gii_path.name}")

                                logger.debug(f"Loading sampling mesh: {sampling_gii_path.name}")
                                sampling_mesh = gifti_to_trimesh(str(sampling_gii_path))
                                if sampling_mesh.is_empty: raise ValueError(f"Sampling mesh {sampling_gii_path.name} is empty.")

                                logger.debug(f"Coloring sampling mesh ({sampling_surf_type_arg})...")
                                project_param_to_surface(mesh=sampling_mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)

                                logger.debug(f"Copying colors from sampling mesh to source mesh ({key})...")
                                if len(sampling_mesh.vertices) != len(source_mesh.vertices): raise ValueError(f"Vertex count mismatch between sampling ({len(sampling_mesh.vertices)}) and source ({len(source_mesh.vertices)})")
                                copy_vertex_colors(sampling_mesh, source_mesh)
                            else:
                                logger.debug(f"Coloring source mesh {key} directly...")
                                project_param_to_surface(mesh=source_mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)
                            
                            # Copy colors from source to target mesh
                            if len(source_mesh.vertices) != len(target_mesh.vertices): raise ValueError(f"Vertex count mismatch between source ({len(source_mesh.vertices)}) and target ({len(target_mesh.vertices)})")
                            copy_vertex_colors(source_mesh, target_mesh)
                            runlog["steps"].append(f"Applied coloring to {key} in source space and warped to target")
                        else:  # color_in == "target"
                            # Color directly in target space
                            if use_cross_sampling:
                                sampling_bids_suffix = SURF_ARG_TO_BIDS_SUFFIX.get(sampling_surf_type_arg)
                                if not sampling_bids_suffix: raise ValueError(f"Invalid --color_sampling_surf '{sampling_surf_type_arg}'")
                                logger.debug(f"Finding sampling surface: hemi={target_hemi}, suffix={sampling_bids_suffix}")
                                
                                # Get the sampling surface in T1 space
                                sampling_gii_path_str = flexible_match(base_dir=anat_dir, subject_id=f"sub-{subject_label_clean}", suffix=f"{sampling_bids_suffix}.surf", hemi=f"hemi-{target_hemi}", ext=".gii", run=args.run, session=args.session)
                                sampling_gii_path = Path(sampling_gii_path_str)
                                if not sampling_gii_path.is_file(): raise FileNotFoundError(f"Sampling surface file not found: {sampling_gii_path}")
                                runlog["steps"].append(f"Located sampling surface {sampling_surf_type_arg} for {key}: {sampling_gii_path.name}")

                                # If we're in MNI space, we need to warp the sampling surface to MNI
                                if args.space.upper() == "MNI":
                                    logger.info(f"Warping sampling surface {sampling_surf_type_arg} to MNI space...")
                                    # Get the necessary files for warping
                                    t1_prep = flexible_match(anat_dir, f"sub-{subject_label_clean}", descriptor="preproc", suffix="T1w", ext=".nii.gz", session=args.session, run=args.run, logger=logger)
                                    logger.debug(f"Found T1 reference: {Path(t1_prep).name}")
                                    mni_ref = flexible_match(anat_dir, f"sub-{subject_label_clean}", space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", session=args.session, run=args.run, logger=logger)
                                    logger.debug(f"Found MNI reference: {Path(mni_ref).name}")
                                    mni_to_t1_xfm = flexible_match(anat_dir, f"sub-{subject_label_clean}", descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image", suffix="xfm", ext=".h5", session=args.session, run=args.run, logger=logger)
                                    logger.debug(f"Found MNI->T1 transform: {Path(mni_to_t1_xfm).name}")
                                    
                                    # Create the warp field only once
                                    t1_to_mni_warp = Path(tmp) / f"warp_sampling_{sampling_surf_type_arg}_T1w-to-MNI.nii.gz"
                                    if not t1_to_mni_warp.exists():
                                        logger.info(f"Creating warp field for sampling surface: {t1_to_mni_warp.name}")
                                        create_mrtrix_warp(str(mni_ref), str(t1_prep), str(mni_to_t1_xfm), str(t1_to_mni_warp), str(tmp), args.verbose)
                                    else:
                                        logger.info(f"Using existing warp field: {t1_to_mni_warp.name}")
                                    
                                    # Warp the sampling surface to MNI
                                    sampling_mni_gii = Path(tmp) / f"sampling_{sampling_surf_type_arg}_{target_hemi}_space-MNI.gii"
                                    logger.info(f"Warping sampling surface to MNI: {sampling_mni_gii.name}")
                                    warp_gifti_vertices(str(sampling_gii_path), str(t1_to_mni_warp), str(sampling_mni_gii), args.verbose)
                                    
                                    # Verify the warped file exists
                                    if not sampling_mni_gii.exists():
                                        raise FileNotFoundError(f"Failed to create warped sampling surface: {sampling_mni_gii}")
                                    
                                    # Update the path to use the warped surface
                                    old_path = sampling_gii_path
                                    sampling_gii_path = sampling_mni_gii
                                    logger.info(f"Using warped sampling surface: {sampling_gii_path.name} (was {old_path.name})")
                                    runlog["steps"].append(f"Warped sampling surface {sampling_surf_type_arg} to MNI space: {sampling_gii_path.name}")

                                logger.debug(f"Loading sampling mesh: {sampling_gii_path.name}")
                                sampling_mesh = gifti_to_trimesh(str(sampling_gii_path))
                                if sampling_mesh.is_empty: raise ValueError(f"Sampling mesh {sampling_gii_path.name} is empty.")
                                
                                # Log vertex counts for debugging
                                logger.debug(f"Sampling mesh vertices: {len(sampling_mesh.vertices)}")
                                logger.debug(f"Target mesh vertices: {len(target_mesh.vertices)}")

                                logger.debug(f"Coloring sampling mesh ({sampling_surf_type_arg})...")
                                project_param_to_surface(mesh=sampling_mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)

                                logger.debug(f"Copying colors from sampling mesh to target mesh ({key})...")
                                if len(sampling_mesh.vertices) != len(target_mesh.vertices): 
                                    logger.error(f"Vertex count mismatch: sampling={len(sampling_mesh.vertices)}, target={len(target_mesh.vertices)}")
                                    raise ValueError(f"Vertex count mismatch between sampling ({len(sampling_mesh.vertices)}) and target ({len(target_mesh.vertices)})")
                                copy_vertex_colors(sampling_mesh, target_mesh)
                            else:
                                logger.debug(f"Coloring target mesh {key} directly...")
                                project_param_to_surface(mesh=target_mesh, param_nifti_path=args.param_map, num_colors=args.num_colors, order=args.order, threshold=args.param_threshold)
                            runlog["steps"].append(f"Applied coloring directly to {key} in target space")
                        
                        colored_keys.append(key)
                    except Exception as e_color:
                        logger.warning(f"Failed process/color component {key}: {e_color}", exc_info=args.verbose)
                        runlog["warnings"].append(f"Processing/Coloring failed for {key}: {e_color}")
                        continue
                else:
                    logger.info(f"Skipping coloring for non-cortical/non-cerebellar component: {key}")
                    runlog["steps"].append(f"Skipped coloring for {key} (not cortical/cerebellar)")

                final_meshes_to_combine.append(target_mesh)
                processed_keys.append(key)

            if not final_meshes_to_combine: raise RuntimeError("No mesh components successfully generated/processed.")

            logger.info(f"Combining {len(final_meshes_to_combine)} mesh components...")
            combined_mesh = trimesh.util.concatenate(final_meshes_to_combine)
            runlog["steps"].append(f"Combined {len(final_meshes_to_combine)} components (colored: {', '.join(colored_keys) or 'None'}). Processed: {', '.join(processed_keys)}")
            if combined_mesh.is_empty: raise RuntimeError("Combined mesh is empty after concatenation.")

            subject_label_clean = args.subject_id.replace('sub-', '')
            fname_parts = [f"sub-{subject_label_clean}"]
            if args.session: fname_parts.append(f"ses-{args.session}")
            if args.run: fname_parts.append(f"run-{args.run}")
            fname_parts.append(f"space-{args.space}")
            fname_parts.append(f"preset-{args.preset}")
            map_stem = Path(args.param_map).stem.replace('.nii','').replace('.gz','')
            fname_parts.append(f"map-{map_stem}")
            if args.color_sampling_surf: fname_parts.append(f"sampling-{args.color_sampling_surf}")
            out_fname = "_".join(fname_parts) + "_selectively_colored.obj"
            out_path = out_dir / out_fname

            logger.info(f"Exporting combined mesh to {out_path} (OBJ format)")
            combined_mesh.export(out_path, file_type="obj")
            runlog["output_files"].append(str(out_path))
            runlog["steps"].append(f"Exported combined selectively colored OBJ => {out_path}")

    except Exception as e:
        logger.error(f"Error during preset processing: {e}", exc_info=args.verbose)
        runlog["warnings"].append(f"Execution failed: {e}")
        write_log(runlog, out_dir, base_name=f"color_preset_{args.preset}_failed_log")
        sys.exit(1)

    write_log(runlog, out_dir, base_name=f"color_preset_{args.preset}_log")
    logger.info("Preset processing finished successfully.")


# --------------------------------------------------------------------------- #
# Main dispatcher
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _build_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    L.setLevel(log_level)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)

    if args.subcommand == "direct":
        _run_direct(args, L)
    elif args.subcommand == "preset":
        _run_preset(args, L)
    else:
        L.critical(f"Unknown subcommand: {args.subcommand}")
        sys.exit(1)


if __name__ == "__main__":
    main()

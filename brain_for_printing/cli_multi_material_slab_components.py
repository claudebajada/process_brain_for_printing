#!/usr/bin/env python
# brain_for_printing/cli_multi_material_slab_components.py

"""
Generates distinct, non-overlapping multi-material brain slab components
suitable for 3D printing.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import shutil 

import trimesh
import nibabel as nib
import numpy as np
from typing import Dict, Optional, List, Any 

# --- Local Imports ---
from .io_utils import temp_dir, require_cmds, flexible_match, run_cmd
from .log_utils import get_logger, write_log
from .surfgen_utils import generate_single_brain_mask_surface, generate_brain_surfaces
from .five_tt_utils import run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes, is_vtk_available
from .mesh_utils import volume_to_gifti, gifti_to_trimesh 
from .config_utils import parse_preset 

from .volumetric_utils import (
    regrid_to_resolution,
    mesh_to_partial_volume,
    binarize_volume_file,
    save_numpy_as_nifti,
    vol_subtract_numpy,
    vol_intersect_numpy,
    vol_union_numpy,
    load_nifti_data
)

# --- Module-Level Helper Functions ---

def get_global_bounds(meshes: List[trimesh.Trimesh]) -> Optional[np.ndarray]:
    """Computes the overall bounding box for a list of meshes."""
    if not meshes:
        return None
    valid_meshes = [m for m in meshes if m is not None and not m.is_empty and hasattr(m, 'vertices')]
    if not valid_meshes:
        return None
        
    all_verts = np.vstack([m.vertices for m in valid_meshes])
    if all_verts.shape[0] == 0:
        return None
    return np.array([np.min(all_verts, axis=0), np.max(all_verts, axis=0)])

def create_slab_definition_boxes(global_bounds: np.ndarray, slab_thickness: float, orientation_axis_idx: int, logger: logging.Logger) -> List[trimesh.primitives.Box]:
    """
    Creates a list of trimesh.primitives.Box objects representing each slab's spatial extent.
    """
    slab_boxes = []
    min_coord_slice_axis = global_bounds[0, orientation_axis_idx]
    max_coord_slice_axis = global_bounds[1, orientation_axis_idx]
    
    current_pos = min_coord_slice_axis
    slab_count = 0
    while current_pos < max_coord_slice_axis:
        slab_top = min(current_pos + slab_thickness, max_coord_slice_axis)
        if slab_top <= current_pos + 1e-6: # Add epsilon for float comparison
            logger.debug(f"Slab top {slab_top} not sufficiently greater than current_pos {current_pos}. Stopping slab creation.")
            break

        box_min = global_bounds[0].copy()
        box_max = global_bounds[1].copy()
        
        box_min[orientation_axis_idx] = current_pos
        box_max[orientation_axis_idx] = slab_top
        
        slab_extents = box_max - box_min
        if np.any(slab_extents <= 1e-6): # Check for non-positive extents with tolerance
            logger.warning(f"Skipping slab {slab_count} due to zero or near-zero extents: {slab_extents}")
            current_pos = slab_top
            if np.isclose(current_pos, max_coord_slice_axis) or current_pos > max_coord_slice_axis:
                break
            continue

        slab_center = (box_min + box_max) / 2.0
        try:
            slab_box = trimesh.creation.box(extents=slab_extents, 
                                            transform=trimesh.transformations.translation_matrix(slab_center))
            slab_boxes.append(slab_box)
            logger.debug(f"Created slab def box {slab_count}: center={slab_center}, extents={slab_extents}")
            slab_count +=1
        except Exception as e_box: 
            logger.error(f"Error creating slab definition box {slab_count} (center: {slab_center}, extents: {slab_extents}): {e_box}")

        current_pos = slab_top
        if np.isclose(current_pos, max_coord_slice_axis) or current_pos > max_coord_slice_axis :
            break
            
    return slab_boxes

def vol_to_mesh(volume_path: Path, output_mesh_path: Path, no_smooth: bool, logger: logging.Logger) -> bool:
    """Converts a NIFTI volume to a Trimesh mesh, optionally smooths, and saves as STL."""
    logger.info(f"Converting volume {volume_path.name} to mesh {output_mesh_path.name}")
    try:
        img_check = nib.load(str(volume_path))
        if np.sum(img_check.get_fdata()) == 0:
            logger.info(f"Volume {volume_path.name} is empty. Skipping mesh generation for this component.")
            return False 
    except Exception as e:
        logger.error(f"Could not load volume {volume_path.name} to check if empty: {e}")
        return False

    temp_gii_path = volume_path.with_name(f"{volume_path.stem}_{os.urandom(4).hex()}.surf.gii")
    
    mesh = None # Initialize mesh to None
    try:
        volume_to_gifti(str(volume_path), str(temp_gii_path), level=0.5) 
        if temp_gii_path.exists(): # Check if GIFTI was created before trying to load
            mesh = gifti_to_trimesh(str(temp_gii_path))
        else:
            logger.warning(f"GIFTI file {temp_gii_path.name} was not created from {volume_path.name}.")
            
    except Exception as e_vtg:
        logger.error(f"volume_to_gifti or gifti_to_trimesh failed for {volume_path.name}: {e_vtg}")
    finally:
        if temp_gii_path.exists(): 
            temp_gii_path.unlink(missing_ok=True)

    if mesh is None or mesh.is_empty:
        logger.warning(f"Mesh from {volume_path.name} is empty or failed to load/convert.")
        return False
    
    try:
        if not mesh.is_watertight:
            mesh = mesh.fill_holes() 
            logger.debug(f"Performed fill_holes on {output_mesh_path.name}.")
    except Exception as e_fill:
        logger.warning(f"Attempting to fill_holes on {output_mesh_path.name} failed: {e_fill}. Proceeding.")

    if not no_smooth:
        logger.info(f"Smoothing {output_mesh_path.name}...")
        try:
            mesh_smooth = trimesh.smoothing.filter_taubin(mesh, iterations=10)
            if mesh_smooth and not mesh_smooth.is_empty: 
                 mesh = mesh_smooth
            else: 
                 logger.warning(f"Taubin smoothing resulted in an empty or invalid mesh for {output_mesh_path.name}. Using original mesh.")
        except Exception as e_smooth: 
            logger.warning(f"Taubin smoothing failed for {output_mesh_path.name}: {e_smooth}. Using unsmoothed.")

    try:
        mesh.fix_normals(multibody=True) 
    except Exception as e_norm:
        logger.warning(f"Fixing normals failed for {output_mesh_path.name}: {e_norm}. Mesh might have issues.")

    try:
        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_mesh_path))
        logger.info(f"Exported mesh: {output_mesh_path}")
        return True
    except Exception as e_export:
        logger.error(f"Failed to export mesh {output_mesh_path}: {e_export}")
        return False

# --- Argument Parser ---
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-material brain slab components for 3D printing.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subjects_dir", required=True, help="Main BIDS derivatives directory.")
    parser.add_argument("--subject_id", required=True, help="Subject ID (e.g., sub-01).")
    parser.add_argument("--output_dir", default="./multi_material_slabs", 
                        help="Base output directory for final STL components and intermediate slab volumes.")
    parser.add_argument("--work_dir", default=None, 
                        help="Directory for all intermediate files (surface generation, 5ttgen, voxelization). If None, a temporary one is created under output_dir.")
    parser.add_argument("--session", default=None, help="BIDS session entity.")
    parser.add_argument("--run", default=None, help="BIDS run entity.")
    
    parser.add_argument("--slab_thickness", type=float, default=5.0, help="Thickness of each slab in mm.")
    parser.add_argument("--slab_orientation", choices=["axial", "coronal", "sagittal"], default="axial", 
                        help="Orientation for slicing.")
    
    parser.add_argument("--voxel_resolution", type=float, default=0.5, 
                        help="Voxel size (mm) for the high-resolution grid for voxelization and volumetric operations.")
    parser.add_argument("--no_final_mesh_smoothing", action="store_true", 
                        help="Disable Taubin smoothing on the final component meshes (STLs).")
    
    parser.add_argument("--skip_outer_csf", action="store_true", help="Skip generation of the outer CSF component.")
    parser.add_argument("--pv_threshold", type=float, default=0.5, help="Threshold for binarizing partial volume images from mesh2voxel.")

    parser.add_argument("--no_clean", action="store_true", help="Keep work directory if it was temporary.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed logging.")
    return parser

# --- Main Workflow ---
def main_wf(args: argparse.Namespace, L: logging.Logger, work_dir: Path, stl_output_base_dir: Path, runlog: Dict[str, Any]):
    """Main workflow logic."""
    runlog["steps"].append("Starting multi-material slab component generation.")
    L.info(f"Target voxel resolution for operations: {args.voxel_resolution} mm")
    L.info(f"Slab thickness: {args.slab_thickness}mm, Orientation: {args.slab_orientation}")

    mrtrix_cmds = ["mrgrid", "mesh2voxel"]
    require_cmds(mrtrix_cmds, logger=L)
    runlog["steps"].append(f"Required MRtrix3 commands verified: {', '.join(mrtrix_cmds)}")

    subject_anat_dir = Path(args.subjects_dir) / args.subject_id / "anat"
    
    full_surf_work_dir = work_dir / "00_parent_surface_generation"
    temp_parent_mesh_files_dir = work_dir / "01_parent_surface_files" 
    slab_surf_files_dir = work_dir / "02_surface_slabs" 
    voxelized_slab_pv_dir = work_dir / "03_voxelized_slab_pvs" 
    binarized_slab_key_mask_dir = work_dir / "04_binarized_slab_key_masks"
    final_material_slab_volumes_dir = work_dir / "05_final_material_slab_volumes"
    
    for d_path in [full_surf_work_dir, temp_parent_mesh_files_dir, slab_surf_files_dir, 
                   voxelized_slab_pv_dir, binarized_slab_key_mask_dir, final_material_slab_volumes_dir]:
        d_path.mkdir(parents=True, exist_ok=True)

    L.info("--- Step 1: Generating full parent T1-space surfaces ---")
    parent_meshes_trimesh: Dict[str, Optional[trimesh.Trimesh]] = {} 

    L.info("Generating BrainMask surface...")
    bm_mesh = generate_single_brain_mask_surface(
        args.subjects_dir, args.subject_id, "T1", 0.0, False, args.run, args.session,
        full_surf_work_dir / "bm_work", L, args.verbose)
    if bm_mesh: parent_meshes_trimesh["BrainMask"] = bm_mesh

    L.info("Generating Pial Complex components...")
    pial_cortical_for_complex, pial_other_for_complex, _ = parse_preset("pial_brain") 
    pial_other_explicit = set()
    for item in pial_other_for_complex:
        if item == "cerebellum": pial_other_explicit.add("cerebellum_cortex")
        else: pial_other_explicit.add(item)
    pial_comp_meshes = generate_brain_surfaces(
        args.subjects_dir, args.subject_id, "T1", 
        tuple(pial_cortical_for_complex), list(pial_other_explicit),       
        [], [], args.run, args.session, args.verbose, str(full_surf_work_dir / "pial_complex_work"))
    for k, v_mesh in pial_comp_meshes.items():
        if v_mesh: parent_meshes_trimesh[k] = v_mesh 

    L.info("Generating White Complex components...")
    white_cortical_for_complex, white_other_for_complex, _ = parse_preset("white_brain")
    white_other_explicit = set()
    for item in white_other_for_complex:
        if item == "cerebellum_wm": white_other_explicit.add("cerebellum_wm") 
        elif item == "cerebellum": 
            L.warning("'cerebellum' found in white_brain preset components, using 'cerebellum_wm'.")
            white_other_explicit.add("cerebellum_wm")
        else: white_other_explicit.add(item)
    white_comp_meshes = generate_brain_surfaces(
        args.subjects_dir, args.subject_id, "T1", 
        tuple(white_cortical_for_complex), list(white_other_explicit),      
        [], [], args.run, args.session, args.verbose, str(full_surf_work_dir / "white_complex_work"))
    for k, v_mesh in white_comp_meshes.items():
        if v_mesh: parent_meshes_trimesh[k] = v_mesh

    if is_vtk_available():
        fs_input_dir = Path(args.subjects_dir) / "sourcedata" / "freesurfer" / args.subject_id
        if fs_input_dir.is_dir():
            five_tt_gen_dir = full_surf_work_dir / "5ttgen_work_multimat" 
            five_tt_gen_dir.mkdir(parents=True, exist_ok=True)
            if run_5ttgen_hsvs_save_temp_bids(args.subject_id, str(fs_input_dir), str(five_tt_gen_dir), args.session, verbose=args.verbose):
                vtk_meshes = load_subcortical_and_ventricle_meshes(str(five_tt_gen_dir))
                parent_meshes_trimesh.update(vtk_meshes) 
            else: L.warning("5ttgen command failed, SGM/Ventricles might be missing.")
        else: L.warning(f"FreeSurfer dir for 5ttgen not found: {fs_input_dir}, skipping SGM/Ventricles.")
    else: L.warning("VTK not available, skipping SGM/Ventricles.")
    
    valid_parent_meshes_trimesh = {k:v for k,v in parent_meshes_trimesh.items() if v and not v.is_empty}
    if not valid_parent_meshes_trimesh: 
        L.error("No valid parent Trimesh surfaces generated. Aborting.")
        runlog["warnings"].append("No parent Trimesh surfaces generated.")
        return False
    L.info(f"Generated {len(valid_parent_meshes_trimesh)} initial parent Trimesh surfaces: {list(valid_parent_meshes_trimesh.keys())}")
    runlog["steps"].append(f"Generated {len(valid_parent_meshes_trimesh)} parent Trimesh surfaces.")

    parent_mesh_files: Dict[str, Path] = {}
    for name, mesh_obj in valid_parent_meshes_trimesh.items():
        safe_name = name.replace("_", "-").replace(" ", "-")
        f_path = temp_parent_mesh_files_dir / f"{args.subject_id}_desc-{safe_name}_parent.obj"
        try:
            mesh_obj.export(str(f_path))
            parent_mesh_files[name] = f_path
        except Exception as e_export:
            L.error(f"Failed to export parent mesh {name} to {f_path}: {e_export}")
    
    if not parent_mesh_files: 
        L.error("Failed to export any parent meshes to files. Aborting.")
        runlog["warnings"].append("Failed to export parent meshes.")
        return False

    L.info(f"--- Step 2: Defining slab geometry from global bounds ---")
    orientation_map = {"axial": 2, "coronal": 1, "sagittal": 0} 
    slice_axis_idx = orientation_map[args.slab_orientation]
    all_meshes_for_bounds_list = list(v for v in valid_parent_meshes_trimesh.values() if v is not None) 
    global_bounds_np_arr = get_global_bounds(all_meshes_for_bounds_list) 
    if global_bounds_np_arr is None: 
        L.error("Could not determine global bounds for slicing. Aborting.")
        runlog["warnings"].append("Failed to get global bounds for slicing.")
        return False
    L.info(f"Global bounds for slicing: Min={global_bounds_np_arr[0]}, Max={global_bounds_np_arr[1]}")
    slab_definition_boxes_list = create_slab_definition_boxes(global_bounds_np_arr, args.slab_thickness, slice_axis_idx, L) 
    num_slabs = len(slab_definition_boxes_list)
    L.info(f"Defined {num_slabs} slab intervals based on global bounds.")
    if num_slabs == 0: 
        L.error("No slab intervals defined. Check slab thickness and mesh bounds. Aborting.")
        runlog["warnings"].append("No slab intervals defined.")
        return False
    runlog["num_slabs"] = num_slabs

    L.info(f"--- Step 3: Preparing high-resolution template ({args.voxel_resolution}mm) ---")
    t1w_native_image_path_str: Optional[str] = None 
    try:
        t1w_native_image_path_str = flexible_match( 
            base_dir=(Path(args.subjects_dir) / args.subject_id / "anat"), 
            subject_id=args.subject_id, descriptor="preproc", suffix="T1w", ext=".nii.gz",
            session=args.session, run=args.run, logger=L )
    except FileNotFoundError:
        L.error(f"T1w preproc image not found for {args.subject_id}. Cannot create voxelization template.")
        runlog["warnings"].append(f"T1w preproc for template not found for {args.subject_id}")
        return False
    
    high_res_template_path = work_dir / f"{args.subject_id}_template_hires_{args.voxel_resolution}mm.nii.gz"
    if not regrid_to_resolution(Path(t1w_native_image_path_str), high_res_template_path, args.voxel_resolution, L, args.verbose): # Path() conversion
        L.error("Failed to create high-resolution template. Aborting.")
        runlog["warnings"].append("Failed to create high_res_template.")
        return False
    runlog["steps"].append(f"High-res template created: {high_res_template_path.name}")
    try:
        template_nifti_image_obj = nib.load(str(high_res_template_path)) 
    except Exception as e_load_template:
        L.error(f"Failed to load created high-res template {high_res_template_path}: {e_load_template}"); 
        runlog["warnings"].append(f"Failed to load high_res_template: {high_res_template_path}")
        return False

    for i in range(num_slabs):
        L.info(f"--- >>> Processing Slab {i+1}/{num_slabs} <<< ---")
        slab_box_def_current = slab_definition_boxes_list[i] 
        L.info(f"--- Slab {i}: Slicing parent surfaces and voxelizing surface slab components ---")
        slab_pv_volume_files_dict: Dict[str, Path] = {} 

        for parent_name, parent_mesh_fpath_val in parent_mesh_files.items(): 
            L.debug(f"Slab {i}: Processing parent surface part: {parent_name} from file {parent_mesh_fpath_val.name}")
            parent_mesh_trimesh_obj = valid_parent_meshes_trimesh.get(parent_name) 
            if not parent_mesh_trimesh_obj: 
                L.warning(f"Trimesh object for {parent_name} (file: {parent_mesh_fpath_val.name}) not found in valid_parent_meshes_trimesh. Skipping.")
                continue

            surface_slab_mesh_obj = None 
            try:
                engine_choice = None 
                if shutil.which('blender'): engine_choice = 'blender'
                
                L.debug(f"Slab {i}: Intersecting {parent_name} with slab box {i} using engine: {engine_choice or 'trimesh_default'}")
                # To ensure intersection works, both meshes should be valid.
                # slab_box_def_current is a trimesh.primitives.Box, which is a Trimesh object.
                if not parent_mesh_trimesh_obj.is_volume: L.debug(f"Parent mesh {parent_name} is not a volume, intersection might be tricky.")
                if not slab_box_def_current.is_volume: L.debug(f"Slab definition box {i} is not a volume.")

                intersected_obj = trimesh.boolean.intersection([parent_mesh_trimesh_obj, slab_box_def_current], engine=engine_choice) 
                if isinstance(intersected_obj, trimesh.Trimesh) and not intersected_obj.is_empty:
                    surface_slab_mesh_obj = intersected_obj
                elif isinstance(intersected_obj, trimesh.Scene):
                    geoms = [g for g in intersected_obj.geometry.values() if isinstance(g, trimesh.Trimesh) and not g.is_empty]
                    if geoms: surface_slab_mesh_obj = trimesh.util.concatenate(geoms)
                    if surface_slab_mesh_obj and surface_slab_mesh_obj.is_empty: surface_slab_mesh_obj = None
                if not surface_slab_mesh_obj:
                    L.debug(f"Slab {i}: Intersection for {parent_name} with slab box was empty or failed.")
                    continue
            except Exception as e_intersect:
                L.warning(f"Slab {i}: Boolean intersection for {parent_name} failed: {e_intersect}", exc_info=args.verbose)
                continue
            
            safe_parent_name_slab = parent_name.replace('_', '-').replace(' ', '-') 
            temp_surface_slab_fpath = slab_surf_files_dir / f"{args.subject_id}_slab-{i}_desc-{safe_parent_name_slab}_surfpart.obj" 
            try:
                surface_slab_mesh_obj.export(str(temp_surface_slab_fpath))
            except Exception as e_export_slab:
                L.warning(f"Slab {i}: Failed to export surface slab {temp_surface_slab_fpath.name}: {e_export_slab}")
                continue

            pv_nifti_fpath = voxelized_slab_pv_dir / f"{args.subject_id}_slab-{i}_desc-{safe_parent_name_slab}_pvol.nii.gz" 
            if mesh_to_partial_volume(temp_surface_slab_fpath, high_res_template_path, pv_nifti_fpath, L, args.verbose):
                slab_pv_volume_files_dict[parent_name] = pv_nifti_fpath
            else:
                L.warning(f"Slab {i}: Failed to voxelize surface slab for {parent_name} from {temp_surface_slab_fpath.name}")
            
            if not args.no_clean: temp_surface_slab_fpath.unlink(missing_ok=True)
        
        if not slab_pv_volume_files_dict: L.warning(f"Slab {i}: No PVs generated. Skipping this slab."); continue
        runlog["steps"].append(f"Slab {i}: Voxelization of {len(slab_pv_volume_files_dict)} surface parts completed.")

        L.info(f"--- Slab {i}: Preparing binarized key structural masks ---")
        M_KeyStruct_Slab_i_data_dict: Dict[str, Optional[np.ndarray]] = {} 

        def load_binarize_save_pv_slab(orig_parent_name: str, pv_paths: Dict[str, Path], 
                                       bin_out_dir: Path, slab_idx: int, 
                                       subj_id: str, thresh: float) -> Optional[np.ndarray]:
            pv_path = pv_paths.get(orig_parent_name)
            safe_orig_name = orig_parent_name.replace('_', '-').replace(' ', '-')
            if not pv_path or not pv_path.exists():
                L.debug(f"PV for '{orig_parent_name}' (safe: {safe_orig_name}) slab {slab_idx} not found. Cannot binarize.")
                return None
            bin_fname = f"{subj_id}_slab-{slab_idx}_desc-{safe_orig_name}_mask.nii.gz" 
            bin_fpath = bin_out_dir / bin_fname 
            if binarize_volume_file(pv_path, bin_fpath, thresh, L):
                loaded_data = load_nifti_data(bin_fpath, L)
                if loaded_data is None: L.error(f"Failed to reload binarized mask {bin_fpath}")
                return loaded_data
            return None

        M_KeyStruct_Slab_i_data_dict["BrainMask"] = load_binarize_save_pv_slab(
            "BrainMask", slab_pv_volume_files_dict, binarized_slab_key_mask_dir, i, args.subject_id, args.pv_threshold)
        if M_KeyStruct_Slab_i_data_dict["BrainMask"] is None:
            L.error(f"BrainMask volume for slab {i} essential and missing. Skipping slab."); continue
        
        # Define the actual keys expected from parent_meshes_trimesh based on how they were added
        pial_complex_actual_keys = ["pial_L", "pial_R", "corpus_callosum", "cerebellum_cortex", "brainstem"]
        pial_arrays_to_union_list = [load_binarize_save_pv_slab(k, slab_pv_volume_files_dict, binarized_slab_key_mask_dir, i, args.subject_id, args.pv_threshold) 
                                 for k in pial_complex_actual_keys if k in slab_pv_volume_files_dict] 
        M_KeyStruct_Slab_i_data_dict["PialComplex"] = vol_union_numpy(pial_arrays_to_union_list)
        if M_KeyStruct_Slab_i_data_dict["PialComplex"] is not None:
            save_numpy_as_nifti(M_KeyStruct_Slab_i_data_dict["PialComplex"], template_nifti_image_obj, binarized_slab_key_mask_dir / f"{args.subject_id}_slab-{i}_desc-PialComplex_mask.nii.gz", L)
        else: M_KeyStruct_Slab_i_data_dict["PialComplex"] = np.zeros_like(M_KeyStruct_Slab_i_data_dict["BrainMask"]) 

        white_complex_actual_keys = ["white_L", "white_R", "corpus_callosum", "cerebellum_wm", "brainstem"] 
        white_arrays_to_union_list = [load_binarize_save_pv_slab(k, slab_pv_volume_files_dict, binarized_slab_key_mask_dir, i, args.subject_id, args.pv_threshold)
                                 for k in white_complex_actual_keys if k in slab_pv_volume_files_dict]
        M_KeyStruct_Slab_i_data_dict["WhiteComplex"] = vol_union_numpy(white_arrays_to_union_list)
        if M_KeyStruct_Slab_i_data_dict["WhiteComplex"] is not None:
            save_numpy_as_nifti(M_KeyStruct_Slab_i_data_dict["WhiteComplex"], template_nifti_image_obj, binarized_slab_key_mask_dir / f"{args.subject_id}_slab-{i}_desc-WhiteComplex_mask.nii.gz", L)
        else: M_KeyStruct_Slab_i_data_dict["WhiteComplex"] = np.zeros_like(M_KeyStruct_Slab_i_data_dict["BrainMask"])

        vent_keys_for_slab_list = [k for k in slab_pv_volume_files_dict if k.startswith("ventricle-")] 
        vent_arrays_to_union_list = [load_binarize_save_pv_slab(k, slab_pv_volume_files_dict, binarized_slab_key_mask_dir, i, args.subject_id, args.pv_threshold) for k in vent_keys_for_slab_list]
        M_KeyStruct_Slab_i_data_dict["VentriclesCombined"] = vol_union_numpy(vent_arrays_to_union_list)
        if M_KeyStruct_Slab_i_data_dict["VentriclesCombined"] is not None:
             save_numpy_as_nifti(M_KeyStruct_Slab_i_data_dict["VentriclesCombined"], template_nifti_image_obj, binarized_slab_key_mask_dir / f"{args.subject_id}_slab-{i}_desc-VentriclesCombined_mask.nii.gz", L)
        else: M_KeyStruct_Slab_i_data_dict["VentriclesCombined"] = np.zeros_like(M_KeyStruct_Slab_i_data_dict["BrainMask"])

        sgm_keys_for_slab_list = [k for k in slab_pv_volume_files_dict if k.startswith("subcortical-")] 
        sgm_arrays_to_union_list = [load_binarize_save_pv_slab(k, slab_pv_volume_files_dict, binarized_slab_key_mask_dir, i, args.subject_id, args.pv_threshold) for k in sgm_keys_for_slab_list]
        M_KeyStruct_Slab_i_data_dict["SGMCombined"] = vol_union_numpy(sgm_arrays_to_union_list)
        if M_KeyStruct_Slab_i_data_dict["SGMCombined"] is not None:
            save_numpy_as_nifti(M_KeyStruct_Slab_i_data_dict["SGMCombined"], template_nifti_image_obj, binarized_slab_key_mask_dir / f"{args.subject_id}_slab-{i}_desc-SGMCombined_mask.nii.gz", L)
        else: M_KeyStruct_Slab_i_data_dict["SGMCombined"] = np.zeros_like(M_KeyStruct_Slab_i_data_dict["BrainMask"])
        
        runlog["steps"].append(f"Slab {i}: Binarized key structural masks prepared.")

        L.info(f"--- Slab {i}: Performing volumetric math for final materials ---")
        m_bm = M_KeyStruct_Slab_i_data_dict["BrainMask"]
        m_pc = M_KeyStruct_Slab_i_data_dict["PialComplex"]
        m_wc = M_KeyStruct_Slab_i_data_dict["WhiteComplex"]
        m_vent = M_KeyStruct_Slab_i_data_dict["VentriclesCombined"]
        m_sgm = M_KeyStruct_Slab_i_data_dict["SGMCombined"]

        # Ensure all arrays are valid before proceeding with math
        if any(v is None for v in [m_bm, m_pc, m_wc, m_vent, m_sgm]): 
            L.error(f"Slab {i}: One or more key masks (BrainMask, PialComplex, WhiteComplex, VentriclesCombined, SGMCombined) is missing or failed to load after binarization. Skipping material definition for this slab.")
            runlog["warnings"].append(f"Slab {i}: Missing key masks for volumetric math.")
            continue

        final_material_volumes_to_mesh_dict: Dict[str, Path] = {} 

        if not args.skip_outer_csf:
            vol_outer_csf = vol_subtract_numpy(m_bm, m_pc)
            p = final_material_slab_volumes_dir / f"{args.subject_id}_slab-{i}_desc-OuterCSF_material.nii.gz"
            if save_numpy_as_nifti(vol_outer_csf, template_nifti_image_obj, p, L): final_material_volumes_to_mesh_dict["OuterCSF"] = p
        
        vol_gm = vol_subtract_numpy(m_pc, m_wc)
        p = final_material_slab_volumes_dir / f"{args.subject_id}_slab-{i}_desc-GreyMatter_material.nii.gz"
        if save_numpy_as_nifti(vol_gm, template_nifti_image_obj, p, L): final_material_volumes_to_mesh_dict["GreyMatter"] = p

        working_wm = m_wc.copy()
        vol_ventricles = vol_intersect_numpy(m_vent, working_wm)
        p = final_material_slab_volumes_dir / f"{args.subject_id}_slab-{i}_desc-Ventricles_material.nii.gz"
        if save_numpy_as_nifti(vol_ventricles, template_nifti_image_obj, p, L): final_material_volumes_to_mesh_dict["Ventricles"] = p
        working_wm = vol_subtract_numpy(working_wm, vol_ventricles) 
        
        vol_sgm = vol_intersect_numpy(m_sgm, working_wm)
        p = final_material_slab_volumes_dir / f"{args.subject_id}_slab-{i}_desc-SubcorticalGrey_material.nii.gz"
        if save_numpy_as_nifti(vol_sgm, template_nifti_image_obj, p, L): final_material_volumes_to_mesh_dict["SubcorticalGrey"] = p
        working_wm = vol_subtract_numpy(working_wm, vol_sgm) 
        
        vol_wm = working_wm 
        p = final_material_slab_volumes_dir / f"{args.subject_id}_slab-{i}_desc-WhiteMatter_material.nii.gz"
        if save_numpy_as_nifti(vol_wm, template_nifti_image_obj, p, L): final_material_volumes_to_mesh_dict["WhiteMatter"] = p
        runlog["steps"].append(f"Slab {i}: Volumetric material definition completed.")

        L.info(f"--- Slab {i}: Meshing final material slab volumes to STL ---")
        material_vol_names = ["OuterCSF", "GreyMatter", "WhiteMatter", "Ventricles", "SubcorticalGrey"]
        if args.skip_outer_csf and "OuterCSF" in material_vol_names: material_vol_names.remove("OuterCSF")

        for mat_name in material_vol_names:
            vol_path = final_material_volumes_to_mesh_dict.get(mat_name) 
            if vol_path and vol_path.exists(): 
                stl_path = stl_output_base_dir / f"{args.subject_id}_slab-{i}_desc-{mat_name}_material.stl"
                if vol_to_mesh(vol_path, stl_path, args.no_final_mesh_smoothing, L):
                    runlog["output_files"].append(str(stl_path))
            else:
                L.warning(f"Final material volume for {mat_name} slab {i} not found or not saved. Skipping mesh.")
        runlog["steps"].append(f"Slab {i}: Meshing of final materials completed.")

    L.info("Multi-material slab component generation workflow finished successfully.")
    return True


def main():
    args = _build_parser().parse_args()
    script_name_stem = Path(__file__).stem
    L_main = get_logger(script_name_stem, level=logging.DEBUG if args.verbose else logging.INFO)

    runlog: Dict[str, Any] = { 
        "tool": script_name_stem,
        "args": {k: v for k, v in vars(args).items() if v is not None},
        "steps": [], "warnings": [],
        "output_dir": os.path.abspath(args.output_dir),
        "output_files": []
    }

    final_stl_output_dir = Path(args.output_dir) 
    final_stl_output_dir.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_dir_path = Path(args.work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
        L_main.info(f"Using specified work directory: {work_dir_path}")
        runlog["steps"].append(f"Using specified work directory: {work_dir_path}")
        success = main_wf(args, L_main, work_dir_path, final_stl_output_dir, runlog)
        # --no_clean is effectively true if --work_dir is specified, as we don't delete it.
        if args.no_clean : L_main.debug("--no_clean is active (or implicit due to --work_dir).")
    else:
        with temp_dir(tag=f"{script_name_stem}_work", keep=args.no_clean, base_dir=str(args.output_dir)) as temp_d_str:
            temp_work_dir_path = Path(temp_d_str)
            L_main.info(f"Using temporary work directory: {temp_work_dir_path}")
            runlog["steps"].append(f"Using temporary work directory: {temp_work_dir_path}")
            success = main_wf(args, L_main, temp_work_dir_path, final_stl_output_dir, runlog)
            if args.no_clean: 
                 runlog["warnings"].append(f"Temporary work directory retained: {temp_work_dir_path}")

    write_log(runlog, str(final_stl_output_dir), base_name=f"{script_name_stem}_log")

    if success:
        L_main.info("Multi-material slab component generation finished successfully.")
    else:
        L_main.error("Multi-material slab component generation failed. See logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

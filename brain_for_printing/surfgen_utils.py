"""
surfgen_utils.py
---------------
Core utilities for generating brain surfaces in different spaces.
Includes functions for generating and warping surfaces.
"""

import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Set
import sys

import trimesh
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


from .io_utils import run_cmd, flexible_match, require_cmd
from .mesh_utils import gifti_to_trimesh, volume_to_gifti
from .warp_utils import create_mrtrix_warp, warp_gifti_vertices
from . import constants as const
from .aseg_utils import extract_structure_surface
from .aseg_utils import convert_fs_aseg_to_t1w


L = logging.getLogger(__name__)

def generate_single_brain_mask_surface(
    subjects_dir: str,
    subject_id: str,
    space: str,
    inflate_mm: float, # Will be called with 0.0 from CLI for no inflation
    no_smooth: bool,   # Will be called with False from CLI for smoothing ON
    run: Optional[str],
    session: Optional[str],
    tmp_dir: Union[str, Path],
    logger: logging.Logger,
    verbose: bool = False
) -> Optional[trimesh.Trimesh]:
    """
    Generates a single surface mesh from a subject's brain mask.
    Inflation and smoothing are controlled by parameters.

    Args:
        subjects_dir: Path to the BIDS derivatives directory.
        subject_id: Subject identifier (e.g., "sub-01").
        space: Target space for the surface ("T1" or "MNI").
        inflate_mm: Amount (in mm) to dilate the binary mask before meshing.
        no_smooth: If True, Taubin smoothing will be skipped.
        run: BIDS run identifier (optional).
        session: BIDS session identifier (optional).
        tmp_dir: Path to the temporary working directory.
        logger: Logger instance.
        verbose: Enable verbose logging for commands.

    Returns:
        Optional[trimesh.Trimesh]: The generated Trimesh object, or None on failure.
    """
    logger.info(f"--- Starting brain mask surface generation for {subject_id} in {space} space ---")
    logger.info(f"Parameters: inflate_mm={inflate_mm}, no_smooth={no_smooth}")
    tmp_path = Path(tmp_dir) 
    tmp_path.mkdir(parents=True, exist_ok=True) # Ensure work sub-directory exists

    try:
        anat_dir = Path(subjects_dir) / subject_id / "anat"
        if not anat_dir.exists():
            logger.error(f"Anatomical directory not found: {anat_dir}")
            return None

        logger.info("Locating T1w brain mask...")
        mask_file_t1_str = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor="desc-brain",
            suffix="mask",
            session=session,
            run=run,
            ext=".nii.gz",
            logger=logger
        )
        logger.info(f"Found T1w brain mask: {mask_file_t1_str}")

        final_mask_path = Path(mask_file_t1_str)

        if space.upper() == "MNI":
            logger.info("Warping brain mask to MNI space...")
            require_cmd("antsApplyTransforms", "https://github.com/ANTsX/ANTs", logger=logger)

            xfm_t1_to_mni_str = flexible_match(
                anat_dir, subject_id,
                descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image",
                suffix="xfm", session=session, run=run, ext=".h5", logger=logger
            )
            mni_template_ref_str = flexible_match(
                anat_dir, subject_id, 
                space="MNI152NLin2009cAsym",
                suffix="T1w", 
                session=session, run=run, ext=".nii.gz", logger=logger
            )
            logger.info(f"Using T1w-to-MNI transform: {xfm_t1_to_mni_str}")
            logger.info(f"Using MNI template reference: {mni_template_ref_str}")

            warped_mask_mni_path = tmp_path / f"{subject_id}_space-MNI_desc-brain_mask.nii.gz"
            cmd_warp = [
                "antsApplyTransforms", "-d", "3",
                "-i", str(final_mask_path),
                "-o", str(warped_mask_mni_path),
                "-r", str(mni_template_ref_str),
                "-t", str(xfm_t1_to_mni_str),
                "-n", "NearestNeighbor"
            ]
            run_cmd(cmd_warp, verbose=verbose)
            logger.info(f"Mask warped to MNI: {warped_mask_mni_path}")
            final_mask_path = warped_mask_mni_path
        elif space.upper() != "T1":
            logger.error(f"Unsupported space for brain mask generation: {space}. Only T1 or MNI.")
            return None

        if inflate_mm > 0:
            logger.info(f"Inflating mask by {inflate_mm} mm...")
            nii = nib.load(final_mask_path)
            data = nii.get_fdata() > 0 
            affine = nii.affine
            voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).mean()
            dilation_steps = int(np.ceil(inflate_mm / voxel_size))
            logger.info(f"Voxel size ≈ {voxel_size:.2f}mm, Dilation steps: {dilation_steps}")

            struct_el = generate_binary_structure(3, 1) 
            dilated_data = data
            if dilation_steps > 0: # Ensure dilation only if steps > 0
                for _ in range(dilation_steps):
                    dilated_data = binary_dilation(dilated_data, structure=struct_el, iterations=1)

            inflated_mask_file = tmp_path / f"{Path(final_mask_path).stem}_inflated-{inflate_mm}mm.nii.gz"
            nib.save(nib.Nifti1Image(dilated_data.astype(np.uint8), affine, header=nii.header), inflated_mask_file)
            logger.info(f"Inflated mask saved to: {inflated_mask_file}")
            final_mask_path = inflated_mask_file
        else:
            logger.info("Skipping mask inflation (inflate_mm <= 0).")


        logger.info(f"Converting mask '{final_mask_path.name}' to GIFTI surface...")
        
        # Construct GIFTI path
        gii_surface_path = tmp_path / f"{Path(final_mask_path).stem}.surf.gii"
        
        # Call volume_to_gifti
        volume_to_gifti(str(final_mask_path), str(gii_surface_path), level=0.5)

        # *** ADDED CHECK for file existence ***
        if not gii_surface_path.exists() or gii_surface_path.stat().st_size == 0:
            logger.error(f"volume_to_gifti failed to create or created an empty output: {gii_surface_path}")
            try:
                mask_img_check = nib.load(str(final_mask_path))
                if np.sum(mask_img_check.get_fdata()) == 0:
                    logger.error(f"The input mask '{final_mask_path.name}' to volume_to_gifti was empty.")
            except Exception as e_load_mask:
                logger.error(f"Could not load input mask {final_mask_path.name} to verify if empty: {e_load_mask}")
            return None # Critical failure if GIFTI not created

        logger.info(f"GIFTI surface created: {gii_surface_path}")

        mesh = gifti_to_trimesh(str(gii_surface_path))
        if mesh.is_empty:
            logger.warning(f"Generated mesh from {gii_surface_path.name} is empty or failed to load. Check volume_to_gifti step.") 
            return None

        if not no_smooth:
            logger.info("Applying Taubin smoothing...")
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
            logger.info("Smoothing applied.")
        else:
            logger.info("Skipping smoothing.")

        logger.info("Inverting mesh normals...")
        mesh.invert()
        logger.info("Normals inverted.")

        logger.info(f"--- Brain mask surface generation for {subject_id} successful ---")
        return mesh

    except FileNotFoundError as e:
        logger.error(f"Essential file not found for brain mask generation: {e}")
        return None
    except RuntimeError as e: 
        logger.error(f"Command execution failed during brain mask generation: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during brain mask surface generation: {e}", exc_info=verbose)
        return None


def generate_brain_surfaces(
    subjects_dir: str,
    subject_id: str,
    space: str = "T1",
    surfaces: Tuple[str, ...] = ("pial",),
    extract_structures: Optional[List[str]] = None,
    no_fill_structures: Optional[List[str]] = None, # Retained for custom mode
    no_smooth_structures: Optional[List[str]] = None, # Retained for custom mode
    run: Optional[str] = None,
    session: Optional[str] = None,
    verbose: bool = False,
    tmp_dir: Optional[str] = None,
    preloaded_vtk_meshes: Optional[Dict[str, trimesh.Trimesh]] = None
) -> Dict[str, Optional[trimesh.Trimesh]]:
    """
    Generate brain surfaces in specified space.
    
    Args:
        subjects_dir: Path to subjects directory
        subject_id: Subject ID (e.g., 'sub-01')
        space: Target space ('T1', 'MNI', or target subject ID)
        surfaces: Tuple of surface types to generate
        extract_structures: List of ASEG structures to extract
        no_fill_structures: List of structures to skip hole filling (used in custom mode)
        no_smooth_structures: List of structures to skip smoothing (used in custom mode)
        run: BIDS run ID
        session: BIDS session ID
        verbose: Enable verbose logging
        tmp_dir: Temporary directory for intermediate files
        preloaded_vtk_meshes: Dictionary of preloaded VTK meshes
        
    Returns:
        Dict[str, Optional[trimesh.Trimesh]]: Dictionary of surface names to trimesh objects
    """
    # Initialize parameters
    if extract_structures is None:
        extract_structures = []
    # For custom mode, these lists will be used. For presets, they are not passed from CLI.
    # Ensure they are lists if None for internal use.
    if no_fill_structures is None:
        no_fill_structures = []
    if no_smooth_structures is None:
        no_smooth_structures = []
        
    STRUCTURE_LABEL_MAP_INTERNAL = { # This map is for generate_brain_surfaces itself
        "brainstem": const.BRAINSTEM_LABEL,
        "cerebellum_wm": const.CEREBELLUM_WM_LABELS,
        "cerebellum_cortex": const.CEREBELLUM_CORTEX_LABELS,
        "cerebellum": const.CEREBELLUM_LABELS, # This will be handled by specific call if needed
        "corpus_callosum": const.CORPUS_CALLOSUM_LABELS,
    }
    
    # Validate extract_structures against THIS map
    valid_extract_structures_for_aseg = []
    for s_name in extract_structures:
        if s_name in STRUCTURE_LABEL_MAP_INTERNAL:
            valid_extract_structures_for_aseg.append(s_name)
        else:
            L.warning(f"Structure '{s_name}' not directly known by generate_brain_surfaces's ASEG map. Will be skipped if it relies on aseg_utils.extract_structure_surface with this exact name.")
    
    extract_structures = valid_extract_structures_for_aseg
    
    SURF_NAME_MAP = {
        "pial": "pial",
        "mid": "midthickness",
        "white": "smoothwm",
        "inflated": "inflated"
    }
    processed_surf_types = {s_type for s_type in surfaces if s_type in SURF_NAME_MAP}
    
    target_space_id: Optional[str] = None
    if space.upper() == "T1":
        space_mode = "T1"
    elif space.upper() == "MNI":
        space_mode = "MNI"
    elif space.startswith("sub-"):
        space_mode = "SUBJECT"
        target_space_id = space
    else:
        L.error(f"Invalid space: {space}")
        raise ValueError(f"Invalid space: {space}")
        
    if space_mode != "T1" and "inflated" in processed_surf_types:
        L.error("Inflated surf only T1 space")
        raise ValueError("Inflated surf only T1 space")
        
    local_tmp = False
    temp_dir_obj = None
    if not tmp_dir:
        try:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"_tmp_surfgen_{subject_id}_")
            tmp_dir_str = temp_dir_obj.name
            local_tmp = True
            L.info(f"Created local tmp dir: {tmp_dir_str}")
        except Exception as e:
            L.error(f"Failed create tmp dir: {e}")
            return {}
    else:
        tmp_dir_str = str(tmp_dir)
        Path(tmp_dir_str).mkdir(parents=True, exist_ok=True)
        
    tmp_dir_path = Path(tmp_dir_str)
    subjects_dir_path = Path(subjects_dir)
    source_subject_id_clean = subject_id.replace('sub-', '')
    source_anat_dir = subjects_dir_path / f"sub-{source_subject_id_clean}" / "anat"
    
    result: Dict[str, Optional[trimesh.Trimesh]] = {}
    source_t1_gifti_paths = {}
    source_t1_meshes = {}
    
    for s_type_name in surfaces: # Changed from 's' to 's_type_name' for clarity
        if s_type_name in SURF_NAME_MAP: # Ensure it's a known cortical surface type
            result[f"{s_type_name}_L"] = None
            result[f"{s_type_name}_R"] = None
    for s_struct_name in extract_structures: # Changed from 's' to 's_struct_name'
        result[s_struct_name] = None
    if preloaded_vtk_meshes:
        for k in preloaded_vtk_meshes:
            result[k] = None
            
    L.info(f"--- Step 1: Gathering T1 surfaces for {subject_id} ---")
    try:
        for surf_type in processed_surf_types:
            suffix = SURF_NAME_MAP[surf_type]
            L.debug(f"Locating T1 {surf_type} ({suffix})")
            try:
                lh = flexible_match(
                    source_anat_dir, subject_id, suffix=f"{suffix}.surf", hemi="L",
                    ext=".gii", run=run, session=session, logger=L )
                source_t1_gifti_paths[f"{surf_type}_L"] = lh
                L.debug(f"Found LH {surf_type}: {Path(lh).name}")
                
                rh = flexible_match(
                    source_anat_dir, subject_id, suffix=f"{suffix}.surf", hemi="R",
                    ext=".gii", run=run, session=session, logger=L )
                source_t1_gifti_paths[f"{surf_type}_R"] = rh
                L.debug(f"Found RH {surf_type}: {Path(rh).name}")
            except FileNotFoundError as e: L.critical(f"Missing T1 FS surf: {e}"); raise
                
        aseg_t1_file_path = None 
        if extract_structures: 
            L.info("Locating T1-space segmentation file for structure extraction...")
            try:
                aseg_t1_file_path = flexible_match(
                    base_dir=source_anat_dir, subject_id=subject_id,
                    descriptor="aseg", suffix="dseg", ext=".nii.gz",
                    session=session, run=run, logger=L )
                L.info(f"  Using fMRIPrep ASEG: {Path(aseg_t1_file_path).name}")
            except FileNotFoundError:
                 try:
                      aseg_t1_file_path = flexible_match(
                          base_dir=source_anat_dir, subject_id=subject_id,
                          descriptor="aseg", suffix="dseg", ext=".nii.gz",
                          session=session, run=run, logger=L )
                      L.info(f"  Using fMRIPrep native-space ASEG: {Path(aseg_t1_file_path).name}")
                 except FileNotFoundError:
                    L.info("  fMRIPrep ASEG not found. Attempting to convert FreeSurfer aseg.mgz to T1w space...")
                    try:
                        aseg_t1_file_path_obj = convert_fs_aseg_to_t1w(
                            subjects_dir=str(subjects_dir_path), 
                            subject_id=subject_id, output_dir=str(tmp_dir_path), 
                            session=session, run=run, verbose=verbose )
                        if aseg_t1_file_path_obj:
                            aseg_t1_file_path = str(aseg_t1_file_path_obj)
                            L.info(f"  Successfully converted FS aseg.mgz to T1w: {Path(aseg_t1_file_path).name}")
                        else: raise RuntimeError("Failed to convert FreeSurfer aseg.mgz.")
                    except Exception as e_convert:
                        L.critical(f"Failed to find or convert an ASEG file in T1 space: {e_convert}")
                        raise RuntimeError("No suitable ASEG file found.") from e_convert

            if not aseg_t1_file_path and extract_structures:
                 raise RuntimeError("ASEG file required for structure extraction but not found or generated.")

        if aseg_t1_file_path and extract_structures : 
             for struct_name in extract_structures:
                L.info(f"Extracting ASEG structure '{struct_name}' from: {Path(aseg_t1_file_path).name}")
                gii_path = extract_structure_surface(
                    structure=struct_name, output_dir=str(tmp_dir_path),
                    aseg_file_path=aseg_t1_file_path, subject_id=subject_id, 
                    target_space="T1", verbose=verbose, logger=L )
                if gii_path and Path(gii_path).exists():
                    L.debug(f"ASEG {struct_name} GIFTI: {Path(gii_path).name}")
                    try: mesh = gifti_to_trimesh(gii_path)
                    except Exception as e: L.warning(f"Load {struct_name} GIFTI fail: {e}"); continue
                    if mesh.is_empty: L.warning(f"ASEG {struct_name} empty."); continue
                    # Use the passed no_fill_structures and no_smooth_structures lists
                    if struct_name not in no_fill_structures: 
                        L.debug(f"Filling {struct_name}")
                        trimesh.repair.fill_holes(mesh)
                    if struct_name not in no_smooth_structures:
                        L.debug(f"Smoothing {struct_name}")
                        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
                    mesh.fix_normals()
                    source_t1_meshes[struct_name] = mesh
                    L.info(f"Processed ASEG {struct_name} -> Trimesh.")
                else: L.warning(f"Failed to generate GIFTI for ASEG '{struct_name}'.")

    except FileNotFoundError as e:
        L.critical(f"Failed to find essential T1 source file: {e}")
        if temp_dir_obj: temp_dir_obj.cleanup()
        sys.exit(1)
    except Exception as e:
        L.error(f"Unexpected error during Step 1 (T1 surface gathering): {e}", exc_info=verbose)
        if temp_dir_obj: temp_dir_obj.cleanup()
        sys.exit(1)

    L.info(f"--- Step 1 Done: {len(source_t1_gifti_paths)} cortical paths, {len(source_t1_meshes)} structure meshes ---")
    
    L.info(f"--- Step 2: Processing for target space: {space_mode} ---")
    
    if space_mode == "T1":
        L.info("Target=T1. Using native.")
        for key, path in source_t1_gifti_paths.items():
            if key not in result or result[key] is None:
                try: L.debug(f"Load T1 {key}"); result[key] = gifti_to_trimesh(path)
                except Exception as e: L.warning(f"Failed load T1 {key}: {e}")
        for key, mesh in source_t1_meshes.items(): result[key] = mesh
        if preloaded_vtk_meshes:
            L.info(f"Adding {len(preloaded_vtk_meshes)} T1 VTK meshes.")
            result.update({k: v for k, v in preloaded_vtk_meshes.items() if v and not v.is_empty})
            
    elif space_mode == "MNI":
        L.info("Target=MNI. Warping T1 surfaces...")
        try:
            t1_prep = flexible_match(
                source_anat_dir, subject_id, descriptor="preproc", suffix="T1w",
                ext=".nii.gz", session=session, run=run, logger=L )
            mni_ref = flexible_match(
                source_anat_dir, subject_id, session=session, run=run, space="MNI152NLin2009cAsym",
                res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", logger=L )
            mni_to_t1_xfm = flexible_match(
                source_anat_dir, subject_id, descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image",
                suffix="xfm", ext=".h5", session=session, run=run, logger=L )
            
            t1_to_mni_warp = tmp_dir_path / f"warp_{subject_id}_T1w-to-MNI.nii.gz"
            create_mrtrix_warp( str(mni_ref), str(t1_prep), str(mni_to_t1_xfm),
                str(t1_to_mni_warp), str(tmp_dir_path), verbose )
            
            for key, t1_gii in source_t1_gifti_paths.items():
                mni_gii = tmp_dir_path / f"{key}_space-MNI.gii"
                L.debug(f"Warping {key} -> MNI")
                try:
                    warp_gifti_vertices(t1_gii, str(t1_to_mni_warp), str(mni_gii), verbose)
                    result[key] = gifti_to_trimesh(str(mni_gii))
                    L.info(f"Warped {key} -> MNI.")
                except Exception as e: L.warning(f"Failed warp/load MNI {key}: {e}")
                    
            for key, t1_mesh in source_t1_meshes.items():
                t1_tmp_gii = tmp_dir_path / f"{key}_T1tmp.gii"
                mni_tmp_gii = tmp_dir_path / f"{key}_MNIwarp.gii"
                L.debug(f"Warping ASEG {key} -> MNI")
                try:
                    coords_da = nib.gifti.GiftiDataArray(t1_mesh.vertices.astype(np.float32))
                    faces_da = nib.gifti.GiftiDataArray(t1_mesh.faces.astype(np.int32))
                    nib.save(nib.gifti.GiftiImage(darrays=[coords_da, faces_da]), str(t1_tmp_gii))
                    warp_gifti_vertices(str(t1_tmp_gii), str(t1_to_mni_warp), str(mni_tmp_gii), verbose)
                    result[key] = gifti_to_trimesh(str(mni_tmp_gii))
                    L.info(f"Warped ASEG {key} -> MNI.")
                except Exception as e: L.warning(f"Failed warp/load MNI ASEG {key}: {e}")
                finally: t1_tmp_gii.unlink(missing_ok=True)
                    
            if preloaded_vtk_meshes: L.warning("MNI warp for VTK not implemented. Excluded.")
                
        except FileNotFoundError as e: L.critical(f"MNI warp FileNotFoundError: {e}"); sys.exit(1)
        except Exception as e: L.error(f"MNI warp error: {e}", exc_info=verbose); sys.exit(1)
            
    elif space_mode == "SUBJECT" and target_space_id:
        L.info(f"Target=Subject {target_space_id}. Warping {subject_id} -> {target_space_id} via MNI...")
        target_anat_dir = Path(subjects_dir) / f"sub-{target_space_id.replace('sub-', '')}" / "anat"
        if not target_anat_dir.is_dir(): L.critical(f"Target subject anat not found: {target_anat_dir}"); sys.exit(1)
            
        try:
            s1_t1 = flexible_match(source_anat_dir, subject_id, descriptor="preproc", suffix="T1w", ext=".nii.gz", session=session, run=run, logger=L)
            s1_mni_ref = flexible_match( source_anat_dir, subject_id, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", session=session, run=run, logger=L )
            s1_mni_to_t1_xfm = flexible_match( source_anat_dir, subject_id, descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image", suffix="xfm", ext=".h5", session=session, run=run, logger=L )
            
            s2_t1 = flexible_match( target_anat_dir, target_space_id, descriptor="preproc", suffix="T1w", ext=".nii.gz", logger=L )
            s2_t1_to_mni_xfm = flexible_match( target_anat_dir, target_space_id, descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image", suffix="xfm", ext=".h5", logger=L )
            try: s2_mni_ref = flexible_match( target_anat_dir, target_space_id, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", logger=L )
            except FileNotFoundError: L.warning(f"MNI ref not found for {target_space_id}, using source's MNI ref."); s2_mni_ref = s1_mni_ref
                
            warp_s1_to_mni = tmp_dir_path / f"warp_{subject_id}_to_MNI.nii.gz"
            create_mrtrix_warp(str(s1_mni_ref), str(s1_t1), str(s1_mni_to_t1_xfm), str(warp_s1_to_mni), str(tmp_dir_path), verbose)
            
            warp_mni_to_s2 = tmp_dir_path / f"warp_MNI_to_{target_space_id}.nii.gz"
            create_mrtrix_warp(str(s2_t1), str(s2_mni_ref), str(s2_t1_to_mni_xfm), str(warp_mni_to_s2), str(tmp_dir_path), verbose)
            
            for key, s1_gii_path in source_t1_gifti_paths.items():
                mni_gii_tmp = tmp_dir_path / f"tmp_{subject_id}_{key}_MNI.gii"
                s2_gii_final = tmp_dir_path / f"{key}_space-{target_space_id}.gii"
                try:
                    L.debug(f"Warping {key}: {subject_id} -> MNI")
                    warp_gifti_vertices(s1_gii_path, str(warp_s1_to_mni), str(mni_gii_tmp), verbose)
                    L.debug(f"Warping {key}: MNI -> {target_space_id}")
                    warp_gifti_vertices(str(mni_gii_tmp), str(warp_mni_to_s2), str(s2_gii_final), verbose)
                    result[key] = gifti_to_trimesh(str(s2_gii_final))
                    L.info(f"Warped cortical {key}: {subject_id} -> {target_space_id}.")
                except Exception as e: L.warning(f"Failed to warp cortical {key} to {target_space_id}: {e}")
                finally: mni_gii_tmp.unlink(missing_ok=True)
                    
            for key, s1_mesh_obj in source_t1_meshes.items():
                s1_gii_tmp = tmp_dir_path / f"tmp_{subject_id}_{key}_T1.gii"
                mni_gii_tmp = tmp_dir_path / f"tmp_{subject_id}_{key}_MNI.gii"
                s2_gii_final = tmp_dir_path / f"{key}_space-{target_space_id}_asegwarp.gii"
                try:
                    coords_da = nib.gifti.GiftiDataArray(s1_mesh_obj.vertices.astype(np.float32))
                    faces_da = nib.gifti.GiftiDataArray(s1_mesh_obj.faces.astype(np.int32))
                    nib.save(nib.gifti.GiftiImage(darrays=[coords_da, faces_da]), str(s1_gii_tmp))
                    
                    L.debug(f"Warping ASEG {key}: {subject_id} -> MNI")
                    warp_gifti_vertices(str(s1_gii_tmp), str(warp_s1_to_mni), str(mni_gii_tmp), verbose)
                    L.debug(f"Warping ASEG {key}: MNI -> {target_space_id}")
                    warp_gifti_vertices(str(mni_gii_tmp), str(warp_mni_to_s2), str(s2_gii_final), verbose)
                    result[key] = gifti_to_trimesh(str(s2_gii_final))
                    L.info(f"Warped ASEG {key}: {subject_id} -> {target_space_id}.")
                except Exception as e: L.warning(f"Failed to warp ASEG {key} to {target_space_id}: {e}")
                finally:
                    s1_gii_tmp.unlink(missing_ok=True)
                    mni_gii_tmp.unlink(missing_ok=True)
                    
            if preloaded_vtk_meshes: L.warning(f"Subject warp for VTK meshes not implemented. Excluded.")
                
        except FileNotFoundError as e: L.critical(f"Cannot find file/xfm for subject-to-subject warp: {e}"); sys.exit(1)
        except Exception as e: L.error(f"Subject-to-subject warp error: {e}", exc_info=verbose); sys.exit(1)

    if local_tmp and temp_dir_obj:
        try: temp_dir_obj.cleanup(); L.info(f"Cleaned up local tmp dir: {tmp_dir_str}")
        except Exception as e_clean: L.warning(f"Failed to cleanup local tmp dir {tmp_dir_str}: {e_clean}")
            
    return result

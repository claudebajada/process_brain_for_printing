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

from .io_utils import run_cmd, flexible_match
from .mesh_utils import gifti_to_trimesh
from .warp_utils import create_mrtrix_warp, warp_gifti_vertices
from . import constants as const
from .aseg_utils import extract_structure_surface
from .aseg_utils import convert_fs_aseg_to_t1w

L = logging.getLogger(__name__)

def generate_brain_surfaces(
    subjects_dir: str,
    subject_id: str,
    space: str = "T1",
    surfaces: Tuple[str, ...] = ("pial",),
    extract_structures: Optional[List[str]] = None,
    no_fill_structures: Optional[List[str]] = None,
    no_smooth_structures: Optional[List[str]] = None,
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
        no_fill_structures: List of structures to skip hole filling
        no_smooth_structures: List of structures to skip smoothing
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
    if no_fill_structures is None:
        no_fill_structures = []
    if no_smooth_structures is None:
        no_smooth_structures = []
        
    # Validate structures
    STRUCTURE_LABEL_MAP = {
        "brainstem": const.BRAINSTEM_LABEL,
        "cerebellum_wm": const.CEREBELLUM_WM_LABELS,
        "cerebellum_cortex": const.CEREBELLUM_CORTEX_LABELS,
        "cerebellum": const.CEREBELLUM_LABELS,
        "corpus_callosum": const.CORPUS_CALLOSUM_LABELS,
    }
    valid_extract_structures = [s for s in extract_structures if s in STRUCTURE_LABEL_MAP]
    if len(valid_extract_structures) != len(extract_structures):
        L.warning(f"Ignoring unknown extract_structures: {set(extract_structures) - set(valid_extract_structures)}")
    extract_structures = valid_extract_structures
    
    # Map surface types
    SURF_NAME_MAP = {
        "pial": "pial",
        "mid": "midthickness",
        "white": "smoothwm",
        "inflated": "inflated"
    }
    processed_surf_types = {s_type for s_type in surfaces if s_type in SURF_NAME_MAP}
    
    # Validate space
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
        
    # Check inflated surface compatibility
    if space_mode != "T1" and "inflated" in processed_surf_types:
        L.error("Inflated surf only T1 space")
        raise ValueError("Inflated surf only T1 space")
        
    # Setup temporary directory
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
    
    # Initialize result dictionary
    result: Dict[str, Optional[trimesh.Trimesh]] = {}
    source_t1_gifti_paths = {}
    source_t1_meshes = {}
    
    # Initialize keys
    for s in surfaces:
        result[f"{s}_L"] = None
        result[f"{s}_R"] = None
    for s in extract_structures:
        result[s] = None
    if preloaded_vtk_meshes:
        for k in preloaded_vtk_meshes:
            result[k] = None
            
    # === Step 1: Gather T1 Native Space Surfaces and Structures ===
    L.info(f"--- Step 1: Gathering T1 surfaces for {subject_id} ---")
    try:
        # --- Locate T1 Cortical Surfaces
        for surf_type in processed_surf_types:
            suffix = SURF_NAME_MAP[surf_type]
            L.debug(f"Locating T1 {surf_type} ({suffix})")
            try:
                lh = flexible_match(
                    source_anat_dir,
                    subject_id,
                    suffix=f"{suffix}.surf",
                    hemi="L",
                    ext=".gii",
                    run=run,
                    session=session,
                    logger=L
                )
                source_t1_gifti_paths[f"{surf_type}_L"] = lh
                L.debug(f"Found LH {surf_type}: {Path(lh).name}")
                
                rh = flexible_match(
                    source_anat_dir,
                    subject_id,
                    suffix=f"{suffix}.surf",
                    hemi="R",
                    ext=".gii",
                    run=run,
                    session=session,
                    logger=L
                )
                source_t1_gifti_paths[f"{surf_type}_R"] = rh
                L.debug(f"Found RH {surf_type}: {Path(rh).name}")
            except FileNotFoundError as e:
                L.critical(f"Missing T1 FS surf: {e}")
                raise
                
        # --- Locate ASEG file ONCE before the loop ---
        aseg_t1_file_path = None # Variable to store the located ASEG file path
        if extract_structures: # Only search for ASEG if structures are requested
            L.info("Locating T1-space segmentation file for structure extraction...")
            try:
                # Try finding fmriprep output first (preferred)
                aseg_t1_file_path = flexible_match(
                    base_dir=source_anat_dir, subject_id=subject_id,
                    # Consider adding space='T1w' if that's standard, otherwise omit for flexibility
                    # space="T1w", # Example: Explicitly look for T1w space if possible
                    descriptor="aseg", suffix="dseg", ext=".nii.gz",
                    session=session, run=run, logger=L )
                L.info(f"  Using fMRIPrep ASEG: {Path(aseg_t1_file_path).name}")
            except FileNotFoundError:
                # If not found, try finding without explicit space T1w (might be native)
                 try:
                      aseg_t1_file_path = flexible_match(
                          base_dir=source_anat_dir, subject_id=subject_id,
                          descriptor="aseg", suffix="dseg", ext=".nii.gz",
                          session=session, run=run, logger=L )
                      L.info(f"  Using fMRIPrep native-space ASEG: {Path(aseg_t1_file_path).name}")
                 except FileNotFoundError:
                    L.info("  fMRIPrep ASEG not found. Attempting to convert FreeSurfer aseg.mgz to T1w space...")
                    try:
                        # Use tmp_dir_path which is defined earlier
                        aseg_t1_file_path_obj = convert_fs_aseg_to_t1w(
                            subjects_dir=str(subjects_dir_path), # Pass full path
                            subject_id=subject_id,
                            output_dir=str(tmp_dir_path), # Use temp dir for output
                            session=session, run=run, verbose=verbose )
                        if aseg_t1_file_path_obj:
                            aseg_t1_file_path = str(aseg_t1_file_path_obj)
                            L.info(f"  Successfully converted FS aseg.mgz to T1w: {Path(aseg_t1_file_path).name}")
                        else:
                            # Raise an error here if conversion fails but structures were requested
                            raise RuntimeError("Failed to convert FreeSurfer aseg.mgz, cannot extract structures.")
                    except Exception as e_convert:
                        L.critical(f"Failed to find or convert an ASEG file in T1 space: {e_convert}")
                        # Decide how to handle: raise error or just warn and skip structures?
                        # Raising error is safer if structures are critical.
                        raise RuntimeError("No suitable ASEG file found for structure extraction.") from e_convert

            # If no ASEG file could be found/created, but structures were requested, raise error
            if not aseg_t1_file_path:
                 raise RuntimeError("ASEG file required but not found or generated.")


        # --- Generate T1 ASEG Structures ---
        # MODIFIED: Loop uses the pre-found aseg_t1_file_path
        if aseg_t1_file_path: # Proceed only if we successfully found the ASEG file
             for struct_name in extract_structures:
                L.info(f"Extracting ASEG structure '{struct_name}' from: {Path(aseg_t1_file_path).name}")
                # MODIFIED: Pass aseg_file_path to the function
                gii_path = extract_structure_surface(
                    structure=struct_name,
                    output_dir=str(tmp_dir_path),
                    aseg_file_path=aseg_t1_file_path, # Pass the found path
                    subject_id=subject_id, # Still needed for output filename
                    target_space="T1",     # Indicate space for filename
                    verbose=verbose,
                    logger=L
                )

            
                if gii_path and Path(gii_path).exists():
                    L.debug(f"ASEG {struct_name} GIFTI: {Path(gii_path).name}")
                    try:
                        mesh = gifti_to_trimesh(gii_path)
                    except Exception as e:
                        L.warning(f"Load {struct_name} GIFTI fail: {e}")
                        continue
                    
                    if mesh.is_empty:
                        L.warning(f"ASEG {struct_name} empty.")
                        continue
                        
                    if struct_name not in no_fill_structures:
                        L.debug(f"Filling {struct_name}")
                        trimesh.repair.fill_holes(mesh)
                    
                    if struct_name not in no_smooth_structures:
                        L.debug(f"Smoothing {struct_name}")
                        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
                    
                    mesh.fix_normals()
                    source_t1_meshes[struct_name] = mesh
                    L.info(f"Processed ASEG {struct_name} -> Trimesh.")
                else:
                    L.warning(f"  Failed to generate GIFTI surface for ASEG structure '{struct_name}'.")

    except FileNotFoundError as e:
        L.critical(f"Failed to find essential T1 source file: {e}")
        sys.exit(1)
    except Exception as e:
        L.error(f"Unexpected error during Step 1 (T1 surface gathering): {e}", exc_info=verbose)
        sys.exit(1)

    L.info(f"--- Step 1 Done: {len(source_t1_gifti_paths)} cortical paths, {len(source_t1_meshes)} structure meshes ---")
    
    # Step 2: Process for target space
    L.info(f"--- Step 2: Processing for target space: {space_mode} ---")
    
    if space_mode == "T1":
        L.info("Target=T1. Using native.")
        for key, path in source_t1_gifti_paths.items():
            if key not in result or result[key] is None:
                try:
                    L.debug(f"Load T1 {key}")
                    result[key] = gifti_to_trimesh(path)
                except Exception as e:
                    L.warning(f"Failed load T1 {key}: {e}")
                    
        for key, mesh in source_t1_meshes.items():
            result[key] = mesh
            
        if preloaded_vtk_meshes:
            L.info(f"Adding {len(preloaded_vtk_meshes)} T1 VTK meshes.")
            result.update({k: v for k, v in preloaded_vtk_meshes.items() if v and not v.is_empty})
            
    elif space_mode == "MNI":
        L.info("Target=MNI. Warping T1 surfaces...")
        try:
            # Get necessary files
            t1_prep = flexible_match(
                source_anat_dir,
                subject_id,
                descriptor="preproc",
                suffix="T1w",
                ext=".nii.gz",
                session=session,
                run=run,
                logger=L
            )
            mni_ref = flexible_match(
                source_anat_dir,
                subject_id,
                session=session,
                run=run,
                space="MNI152NLin2009cAsym",
                res="*",
                descriptor="preproc",
                suffix="T1w",
                ext=".nii.gz",
                logger=L
            )
            mni_to_t1_xfm = flexible_match(
                source_anat_dir,
                subject_id,
                descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image",
                suffix="xfm",
                ext=".h5",
                session=session,
                run=run,
                logger=L
            )
            
            # Create warp field
            t1_to_mni_warp = tmp_dir_path / f"warp_{subject_id}_T1w-to-MNI.nii.gz"
            create_mrtrix_warp(
                str(mni_ref),
                str(t1_prep),
                str(mni_to_t1_xfm),
                str(t1_to_mni_warp),
                str(tmp_dir_path),
                verbose
            )
            
            # Warp cortical surfaces
            for key, t1_gii in source_t1_gifti_paths.items():
                mni_gii = tmp_dir_path / f"{key}_space-MNI.gii"
                L.debug(f"Warping {key} -> MNI")
                try:
                    warp_gifti_vertices(t1_gii, str(t1_to_mni_warp), str(mni_gii), verbose)
                    result[key] = gifti_to_trimesh(str(mni_gii))
                    L.info(f"Warped {key} -> MNI.")
                except Exception as e:
                    L.warning(f"Failed warp/load MNI {key}: {e}")
                    
            # Warp ASEG structures
            for key, t1_mesh in source_t1_meshes.items():
                t1_tmp = tmp_dir_path / f"{key}_T1tmp.gii"
                mni_tmp = tmp_dir_path / f"{key}_MNIwarp.gii"
                L.debug(f"Warping ASEG {key} -> MNI")
                try:
                    # Convert mesh to GIFTI
                    coords = nib.gifti.GiftiDataArray(t1_mesh.vertices.astype(np.float32))
                    faces = nib.gifti.GiftiDataArray(t1_mesh.faces.astype(np.int32))
                    nib.save(nib.gifti.GiftiImage(darrays=[coords, faces]), str(t1_tmp))
                    
                    # Warp to MNI
                    warp_gifti_vertices(str(t1_tmp), str(t1_to_mni_warp), str(mni_tmp), verbose)
                    result[key] = gifti_to_trimesh(str(mni_tmp))
                    L.info(f"Warped ASEG {key} -> MNI.")
                except Exception as e:
                    L.warning(f"Failed warp/load MNI ASEG {key}: {e}")
                finally:
                    t1_tmp.unlink(missing_ok=True)
                    
            if preloaded_vtk_meshes:
                L.warning("MNI warp for VTK not implemented. Excluded.")
                
        except FileNotFoundError as e:
            L.critical(f"MNI warp FileNotFoundError: {e}")
            sys.exit(1)
        except Exception as e:
            L.error(f"MNI warp error: {e}", exc_info=verbose)
            sys.exit(1)
            
    elif space_mode == "SUBJECT" and target_space_id:
        L.info(f"Target=Subject {target_space_id}. Warping {subject_id} -> {target_space_id} via MNI...")
        target_anat = Path(subjects_dir) / f"sub-{target_space_id.replace('sub-', '')}" / "anat"
        if not target_anat.is_dir():
            L.critical(f"Target subject anat not found: {target_anat}")
            sys.exit(1)
            
        try:
            # Get source files
            s1_t1 = flexible_match(
                source_anat_dir,
                subject_id,
                descriptor="preproc",
                suffix="T1w",
                ext=".nii.gz",
                session=session,
                run=run,
                logger=L
            )
            s1_mni = flexible_match(
                source_anat_dir,
                subject_id,
                space="MNI152NLin2009cAsym",
                res="*",
                descriptor="preproc",
                suffix="T1w",
                ext=".nii.gz",
                session=session,
                run=run,
                logger=L
            )
            s1_mni2t1_xfm = flexible_match(
                source_anat_dir,
                subject_id,
                descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image",
                suffix="xfm",
                ext=".h5",
                session=session,
                run=run,
                logger=L
            )
            
            # Get target files
            s2_t1 = flexible_match(
                target_anat,
                target_space_id,
                descriptor="preproc",
                suffix="T1w",
                ext=".nii.gz",
                logger=L
            )
            s2_t12mni_xfm = flexible_match(
                target_anat,
                target_space_id,
                descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image",
                suffix="xfm",
                ext=".h5",
                logger=L
            )
            
            try:
                s2_mni = flexible_match(
                    target_anat,
                    target_space_id,
                    space="MNI152NLin2009cAsym",
                    res="*",
                    descriptor="preproc",
                    suffix="T1w",
                    ext=".nii.gz",
                    logger=L
                )
            except FileNotFoundError:
                L.warning(f"MNI ref not found in {target_space_id}, using source's.")
                s2_mni = s1_mni
                
            # Create warp fields
            warp_s1_mni = tmp_dir_path / f"_{subject_id}_to_MNI.nii.gz"
            create_mrtrix_warp(str(s1_mni), str(s1_t1), str(s1_mni2t1_xfm), str(warp_s1_mni), str(tmp_dir_path), verbose)
            
            warp_mni_s2 = tmp_dir_path / f"_MNI_to_{target_space_id}.nii.gz"
            create_mrtrix_warp(str(s2_t1), str(s2_mni), str(s2_t12mni_xfm), str(warp_mni_s2), str(tmp_dir_path), verbose)
            
            # Warp cortical surfaces
            for key, s1_gii in source_t1_gifti_paths.items():
                mni_gii = tmp_dir_path / f"_{subject_id}_{key}_MNI.gii"
                s2_gii = tmp_dir_path / f"{key}_{target_space_id}.gii"
                try:
                    L.debug(f"{key}: S1->MNI")
                    warp_gifti_vertices(s1_gii, str(warp_s1_mni), str(mni_gii), verbose)
                    
                    L.debug(f"{key}: MNI->S2")
                    warp_gifti_vertices(str(mni_gii), str(warp_mni_s2), str(s2_gii), verbose)
                    
                    result[key] = gifti_to_trimesh(str(s2_gii))
                    L.info(f"Warped cortical {key} -> {target_space_id}.")
                except Exception as e:
                    L.warning(f"Failed to warp {key}: {e}")
                    if mni_gii.exists():
                        mni_gii.unlink()
                    if s2_gii.exists():
                        s2_gii.unlink()
                    continue
                    
            # Warp ASEG structures
            for key, s1_mesh in source_t1_meshes.items():
                t1_gii = tmp_dir_path / f"_{subject_id}_{key}_T1.gii"
                mni_gii = tmp_dir_path / f"_{subject_id}_{key}_MNI.gii"
                s2_gii = tmp_dir_path / f"{key}_{target_space_id}warp.gii"
                try:
                    # Convert mesh to GIFTI
                    coords = nib.gifti.GiftiDataArray(s1_mesh.vertices.astype(np.float32))
                    faces = nib.gifti.GiftiDataArray(s1_mesh.faces.astype(np.int32))
                    nib.save(nib.gifti.GiftiImage(darrays=[coords, faces]), str(t1_gii))
                    
                    # First warp to MNI
                    L.debug(f"ASEG {key}: S1->MNI")
                    warp_gifti_vertices(str(t1_gii), str(warp_s1_mni), str(mni_gii), verbose)
                    
                    # Then warp to target subject
                    L.debug(f"ASEG {key}: MNI->S2")
                    warp_gifti_vertices(str(mni_gii), str(warp_mni_s2), str(s2_gii), verbose)
                    
                    result[key] = gifti_to_trimesh(str(s2_gii))
                    L.info(f"Warped ASEG {key} -> {target_space_id}.")
                except Exception as e:
                    L.warning(f"Failed to warp ASEG {key}: {e}")
                    for f in [t1_gii, mni_gii, s2_gii]:
                        if f.exists():
                            f.unlink()
                    continue
                    
            if preloaded_vtk_meshes:
                L.warning(f"Subject warp for VTK not implemented. Excluded.")
                
        except FileNotFoundError as e:
            L.critical(f"Cannot find file/xfm for subject warp: {e}")
            sys.exit(1)
        except Exception as e:
            L.error(f"Subject warp error: {e}", exc_info=verbose)
            sys.exit(1)
            
    return result 

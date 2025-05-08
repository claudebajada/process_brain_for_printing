# brain_for_printing/surfaces.py

"""
surfaces.py
-----------
Generate multiple brain surfaces (cortical + optional brainstem, cerebellum, etc.)
in T1, MNI, or target subject space. Includes function to run
5ttgen hsvs and manage its working directory.
"""
import os
import uuid
import shutil
import trimesh
import logging
import subprocess
import tempfile
import json
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import sys # Keep Added Import

# Import corrected flexible_match
from .io_utils import run_cmd, flexible_match, first_match
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .warp_utils import create_mrtrix_warp, warp_gifti_vertices
from . import constants as const
import numpy as np
import nibabel as nib

# Handle Optional VTK Import
try:
    from vtk.util import numpy_support # type: ignore
    import vtk                     # type: ignore
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

L = logging.getLogger(__name__)

# Define stubs if VTK not available *before* they are potentially used
if not VTK_AVAILABLE:
    def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[object]:
        logger.error(f"VTK unavailable, cannot read: {path}"); return None
    def _vtk_polydata_to_trimesh(poly: Optional[object]) -> Optional[trimesh.Trimesh]:
        log_func = L.error if 'L' in globals() else print; log_func("VTK unavailable."); return None

# --- Mask Extraction Helper ---
def _extract_structure_mask_t1( aseg_nifti_path: str, label_ids: List[int], output_mask_nifti_path: str, verbose: bool = False ):
    if not Path(aseg_nifti_path).exists(): L.error(f"Input NIfTI not found: {aseg_nifti_path}"); return False
    match_str = [str(lbl) for lbl in label_ids]; L.info(f"Extracting {match_str} from {Path(aseg_nifti_path).name} -> {Path(output_mask_nifti_path).name}")
    try: run_cmd(["mri_binarize", "--i", aseg_nifti_path, "--match", *match_str, "--o", output_mask_nifti_path], verbose=verbose)
    except Exception as e: L.error(f"mri_binarize failed: {e}", exc_info=verbose); return False
    output_path = Path(output_mask_nifti_path)
    if not output_path.exists() or output_path.stat().st_size == 0: L.error(f"Output mask empty/not created: {output_path.name}"); return False
    return True

# --- ASEG Structure Surface Extraction ---
def extract_structure_surface( subjects_dir: str, subject_id: str, label_ids: List[int], output_tag: str, space: str = 'T1', tmp_dir: str = '.', verbose: bool = False, session: Optional[str] = None, run: Optional[str] = None ) -> Optional[str]:
    tmp_dir_path = Path(tmp_dir); tmp_dir_path.mkdir(parents=True, exist_ok=True)
    subject_id_clean = subject_id.replace('sub-', '')
    anat_dir = Path(subjects_dir) / f"sub-{subject_id_clean}" / "anat"
    output_mask_nii_path = tmp_dir_path / f"{output_tag}_mask_space-{space}_id-{uuid.uuid4().hex[:4]}.nii.gz"
    output_gii_path = tmp_dir_path / f"{output_tag}_space-{space}.surf.gii"
    aseg_in_target_space: Optional[str] = None
    try:
        if space.upper() == "T1":
            L.info(f"Locating T1-space aseg for {subject_id} ({output_tag})")
            # --- FIX: Use desc-aseg ---
            aseg_in_target_space = flexible_match( base_dir=anat_dir, subject_id=subject_id, descriptor="desc-aseg", suffix="dseg", ext=".nii.gz", session=session, run=run, logger=L)
            L.info(f"Found T1-space aseg: {Path(aseg_in_target_space).name}")
        elif space.upper() == "MNI":
            L.info(f"Preparing MNI-space aseg for {subject_id} ({output_tag})")
            # --- FIX: Use desc-aseg ---
            aseg_t1_path = flexible_match( base_dir=anat_dir, subject_id=subject_id, descriptor="desc-aseg", suffix="dseg", ext=".nii.gz", session=session, run=run, logger=L)
            L.debug(f"Found T1 aseg: {Path(aseg_t1_path).name}")
            xfm_t1_to_mni = flexible_match( base_dir=anat_dir, subject_id=subject_id, descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image", suffix="xfm", ext=".h5", session=session, run=run, logger=L)
            L.debug(f"Found T1->MNI xfm: {Path(xfm_t1_to_mni).name}")
            try: # Find MNI space reference geometry (use dseg if possible)
                mni_ref_path_str = flexible_match( base_dir=anat_dir, subject_id=subject_id, space="MNI152NLin2009cAsym", res="*", descriptor="desc-aseg", suffix="dseg", ext=".nii.gz", session=session, run=run, logger=L)
            except FileNotFoundError:
                 try: mni_ref_path_str = flexible_match( base_dir=anat_dir, subject_id=subject_id, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", session=session, run=run, logger=L)
                 except FileNotFoundError: L.warning(f"MNI ref not found for {subject_id}, using T1 aseg."); mni_ref_path_str = aseg_t1_path
            warped_aseg_path = tmp_dir_path / f"{output_tag}_aseg_in_mni_id-{uuid.uuid4().hex[:4]}.nii.gz"
            L.info(f"Warping {Path(aseg_t1_path).name} -> MNI ({warped_aseg_path.name})")
            run_cmd([ "antsApplyTransforms", "-d", "3", "-i", aseg_t1_path, "-o", str(warped_aseg_path), "-r", mni_ref_path_str, "-t", xfm_t1_to_mni, "-n", "NearestNeighbor" ], verbose=verbose)
            aseg_in_target_space = str(warped_aseg_path)
        else: L.error(f"Unsupported ASEG space: {space}"); return None
    except FileNotFoundError as e: L.error(f"ASEG prep FileNotFoundError: {e}", exc_info=verbose); return None
    except Exception as e: L.error(f"ASEG prep error: {e}", exc_info=verbose); return None
    if not aseg_in_target_space or not Path(aseg_in_target_space).exists(): L.error(f"Aseg target space verified fail: '{aseg_in_target_space}'"); return None
    success = _extract_structure_mask_t1( aseg_in_target_space, label_ids, str(output_mask_nii_path), verbose )
    if not success: L.error(f"Mask creation failed for {output_tag} in {space}."); return None
    try: volume_to_gifti(str(output_mask_nii_path), str(output_gii_path), level=0.5); L.info(f"Created GIFTI: {output_gii_path.name}"); return str(output_gii_path)
    except Exception as e: L.error(f"NIfTI->GIFTI failed for {output_tag}: {e}", exc_info=verbose); return None

# --- Main Surface Generation Function ---
def generate_brain_surfaces( subjects_dir: str, subject_id: str, space: str = "T1", surfaces: Tuple[str, ...] = ("pial",), extract_structures: Optional[List[str]] = None, no_fill_structures: Optional[List[str]] = None, no_smooth_structures: Optional[List[str]] = None, run: Optional[str] = None, session: Optional[str] = None, verbose: bool = False, tmp_dir: Optional[str] = None, preloaded_vtk_meshes: Optional[Dict[str, trimesh.Trimesh]] = None ) -> Dict[str, Optional[trimesh.Trimesh]]:
    # ... (Initial setup as before) ...
    if extract_structures is None: extract_structures = []
    if no_fill_structures is None: no_fill_structures = []
    if no_smooth_structures is None: no_smooth_structures = []
    STRUCTURE_LABEL_MAP = { "brainstem": const.BRAINSTEM_LABEL, "cerebellum_wm": const.CEREBELLUM_WM_LABELS, "cerebellum_cortex": const.CEREBELLUM_CORTEX_LABELS, "cerebellum": const.CEREBELLUM_LABELS, "corpus_callosum": const.CORPUS_CALLOSUM_LABELS, }
    valid_extract_structures = [s for s in extract_structures if s in STRUCTURE_LABEL_MAP];
    if len(valid_extract_structures) != len(extract_structures): L.warning(f"Ignoring unknown extract_structures: {set(extract_structures) - set(valid_extract_structures)}")
    extract_structures = valid_extract_structures
    SURF_NAME_MAP = { "pial": "pial", "mid": "midthickness", "white": "smoothwm", "inflated": "inflated" }
    processed_surf_types = {s_type for s_type in surfaces if s_type in SURF_NAME_MAP}
    target_space_id: Optional[str] = None
    if space.upper() == "T1": space_mode = "T1"
    elif space.upper() == "MNI": space_mode = "MNI"
    elif space.startswith("sub-"): space_mode = "SUBJECT"; target_space_id = space
    else: L.error(f"Invalid space: {space}"); raise ValueError(f"Invalid space: {space}")
    if space_mode != "T1" and "inflated" in processed_surf_types: L.error("Inflated surf only T1 space"); raise ValueError("Inflated surf only T1 space")
    local_tmp = False; temp_dir_obj = None
    if not tmp_dir:
        try: temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"_tmp_surfgen_{subject_id}_"); tmp_dir_str = temp_dir_obj.name; local_tmp = True; L.info(f"Created local tmp dir: {tmp_dir_str}")
        except Exception as e: L.error(f"Failed create tmp dir: {e}"); return {}
    else: tmp_dir_str = str(tmp_dir); Path(tmp_dir_str).mkdir(parents=True, exist_ok=True)
    tmp_dir_path = Path(tmp_dir_str)
    subjects_dir_path = Path(subjects_dir)
    source_subject_id_clean = subject_id.replace('sub-', '')
    source_anat_dir = subjects_dir_path / f"sub-{source_subject_id_clean}" / "anat"
    result: Dict[str, Optional[trimesh.Trimesh]] = {}; source_t1_gifti_paths = {}; source_t1_meshes = {}
    for s in surfaces: result[f"{s}_L"]=None; result[f"{s}_R"]=None # Init keys
    for s in extract_structures: result[s]=None
    if preloaded_vtk_meshes:
        for k in preloaded_vtk_meshes: result[k]=None

    L.info(f"--- Step 1: Gathering T1 surfaces for {subject_id} ---")
    try:
        for surf_type in processed_surf_types:
            suffix = SURF_NAME_MAP[surf_type]; L.debug(f"Locating T1 {surf_type} ({suffix})")
            try:
                lh = flexible_match(source_anat_dir, subject_id, suffix=f"{suffix}.surf", hemi="L", ext=".gii", run=run, session=session, logger=L); source_t1_gifti_paths[f"{surf_type}_L"]=lh; L.debug(f"Found LH {surf_type}: {Path(lh).name}")
                rh = flexible_match(source_anat_dir, subject_id, suffix=f"{suffix}.surf", hemi="R", ext=".gii", run=run, session=session, logger=L); source_t1_gifti_paths[f"{surf_type}_R"]=rh; L.debug(f"Found RH {surf_type}: {Path(rh).name}")
            except FileNotFoundError as e: L.critical(f"Missing T1 FS surf: {e}"); raise
        for struct_name in extract_structures:
            labels = STRUCTURE_LABEL_MAP[struct_name]; L.info(f"Extracting ASEG '{struct_name}' in T1...")
            gii_path = extract_structure_surface( str(subjects_dir_path), subject_id, labels, struct_name, "T1", str(tmp_dir_path), verbose, session, run )
            if gii_path and Path(gii_path).exists():
                L.debug(f"ASEG {struct_name} GIFTI: {Path(gii_path).name}")
                try: mesh = gifti_to_trimesh(gii_path)
                except Exception as e: L.warning(f"Load {struct_name} GIFTI fail: {e}"); continue
                if mesh.is_empty: L.warning(f"ASEG {struct_name} empty."); continue
                if struct_name not in no_fill_structures: L.debug(f"Filling {struct_name}"); trimesh.repair.fill_holes(mesh)
                if struct_name not in no_smooth_structures: L.debug(f"Smoothing {struct_name}"); trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
                mesh.fix_normals(); source_t1_meshes[struct_name] = mesh; L.info(f"Processed ASEG {struct_name} -> Trimesh.")
            else: L.warning(f"Failed generate GIFTI for ASEG {struct_name} in T1.")
    except FileNotFoundError as e: L.critical(f"Failed find critical T1 source: {e}"); sys.exit(1)
    except Exception as e: L.error(f"Error Step 1: {e}", exc_info=verbose); sys.exit(1)
    L.info(f"--- Step 1 Done ({len(source_t1_gifti_paths)} cortical paths, {len(source_t1_meshes)} ASEG) ---")

    L.info(f"--- Step 2: Processing for target space: {space_mode} ---")
    if space_mode == "T1":
        L.info("Target=T1. Using native."); # ... (T1 logic as before) ...
        for key, path in source_t1_gifti_paths.items():
            if key not in result or result[key] is None:
                try: L.debug(f"Load T1 {key}"); result[key] = gifti_to_trimesh(path)
                except Exception as e: L.warning(f"Failed load T1 {key}: {e}")
        for key, mesh in source_t1_meshes.items(): result[key] = mesh
        if preloaded_vtk_meshes: L.info(f"Adding {len(preloaded_vtk_meshes)} T1 VTK meshes."); result.update({k:v for k,v in preloaded_vtk_meshes.items() if v and not v.is_empty})

    elif space_mode == "MNI":
        L.info("Target=MNI. Warping T1 surfaces...");
        try:
            t1_prep = flexible_match( source_anat_dir, subject_id, descriptor="preproc", suffix="T1w", ext=".nii.gz", session=session, run=run, logger=L )
            mni_ref = flexible_match( source_anat_dir, subject_id, session=session, run=run, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", logger=L )
            mni_to_t1_xfm = flexible_match( source_anat_dir, subject_id, descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image", suffix="xfm", ext=".h5", session=session, run=run, logger=L )
            t1_to_mni_warp = tmp_dir_path / f"warp_{subject_id}_T1w-to-MNI.nii.gz"
            create_mrtrix_warp( str(mni_ref), str(t1_prep), str(mni_to_t1_xfm), str(t1_to_mni_warp), str(tmp_dir_path), verbose )
            for key, t1_gii in source_t1_gifti_paths.items():
                mni_gii = tmp_dir_path / f"{key}_space-MNI.gii"; L.debug(f"Warping {key} -> MNI")
                try: warp_gifti_vertices(t1_gii, str(t1_to_mni_warp), str(mni_gii), verbose); result[key] = gifti_to_trimesh(str(mni_gii)); L.info(f"Warped {key} -> MNI.")
                except Exception as e: L.warning(f"Failed warp/load MNI {key}: {e}")
            for key, t1_mesh in source_t1_meshes.items():
                t1_tmp = tmp_dir_path / f"{key}_T1tmp.gii"; mni_tmp = tmp_dir_path / f"{key}_MNIwarp.gii"; L.debug(f"Warping ASEG {key} -> MNI")
                try:
                    coords = nib.gifti.GiftiDataArray(t1_mesh.vertices.astype(np.float32)); faces = nib.gifti.GiftiDataArray(t1_mesh.faces.astype(np.int32))
                    nib.save(nib.gifti.GiftiImage(darrays=[coords, faces]), str(t1_tmp))
                    warp_gifti_vertices(str(t1_tmp), str(t1_to_mni_warp), str(mni_tmp), verbose); result[key] = gifti_to_trimesh(str(mni_tmp)); L.info(f"Warped ASEG {key} -> MNI.")
                except Exception as e: L.warning(f"Failed warp/load MNI ASEG {key}: {e}")
                finally: t1_tmp.unlink(missing_ok=True)
            if preloaded_vtk_meshes: L.warning("MNI warp for VTK not implemented. Excluded.")
        except FileNotFoundError as e: L.critical(f"MNI warp FileNotFoundError: {e}"); sys.exit(1)
        except Exception as e: L.error(f"MNI warp error: {e}", exc_info=verbose); sys.exit(1)

    elif space_mode == "SUBJECT" and target_space_id:
        L.info(f"Target=Subject {target_space_id}. Warping {subject_id} -> {target_space_id} via MNI...");
        target_anat = Path(subjects_dir) / f"sub-{target_space_id.replace('sub-', '')}" / "anat"
        if not target_anat.is_dir(): L.critical(f"Target subject anat not found: {target_anat}"); sys.exit(1)
        try:
            # S1 Files
            s1_t1 = flexible_match( source_anat_dir, subject_id, descriptor="preproc", suffix="T1w", ext=".nii.gz", session=session, run=run, logger=L )
            s1_mni = flexible_match( source_anat_dir, subject_id, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", session=session, run=run, logger=L )
            s1_mni2t1_xfm = flexible_match( source_anat_dir, subject_id, descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image", suffix="xfm", ext=".h5", session=session, run=run, logger=L )
            # S2 Files
            s2_t1 = flexible_match( target_anat, target_space_id, descriptor="preproc", suffix="T1w", ext=".nii.gz", logger=L )
            s2_t12mni_xfm = flexible_match( target_anat, target_space_id, descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image", suffix="xfm", ext=".h5", logger=L )
            try: s2_mni = flexible_match( target_anat, target_space_id, space="MNI152NLin2009cAsym", res="*", descriptor="preproc", suffix="T1w", ext=".nii.gz", logger=L )
            except FileNotFoundError: L.warning(f"MNI ref not found in {target_space_id}, using source's."); s2_mni = s1_mni
            # Warps
            warp_s1_mni = tmp_dir_path/f"_{subject_id}_to_MNI.nii.gz"; create_mrtrix_warp(str(s1_mni), str(s1_t1), str(s1_mni2t1_xfm), str(warp_s1_mni), str(tmp_dir_path), verbose)
            warp_mni_s2 = tmp_dir_path/f"_MNI_to_{target_space_id}.nii.gz"; create_mrtrix_warp(str(s2_t1), str(s2_mni), str(s2_t12mni_xfm), str(warp_mni_s2), str(tmp_dir_path), verbose)
            # Apply Cortical
            for key, s1_gii in source_t1_gifti_paths.items():
                mni_gii = tmp_dir_path/f"_{subject_id}_{key}_MNI.gii"
                s2_gii = tmp_dir_path/f"{key}_{target_space_id}.gii"
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
            # Apply ASEG
            for key, s1_mesh in source_t1_meshes.items():
                t1_gii = tmp_dir_path/f"_{subject_id}_{key}_T1.gii"
                mni_gii = tmp_dir_path/f"_{subject_id}_{key}_MNI.gii"
                s2_gii = tmp_dir_path/f"{key}_{target_space_id}warp.gii"
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
            if preloaded_vtk_meshes: L.warning(f"Subject warp for VTK not implemented. Excluded.")
        except FileNotFoundError as e: L.critical(f"Cannot find file/xfm for subject warp: {e}"); sys.exit(1)
        except Exception as e: L.error(f"General error subject warp: {e}", exc_info=verbose); sys.exit(1)
    L.info(f"--- Step 2 Done: Processed for {space_mode} ---")

    if local_tmp and temp_dir_obj:
        try: temp_dir_obj.cleanup(); L.info(f"Removed tmp dir: {tmp_dir_str}")
        except Exception as e: L.warning(f"Failed remove tmp dir {tmp_dir_str}: {e}")
    final_result = {k: v for k, v in result.items() if isinstance(v, trimesh.Trimesh) and not v.is_empty}
    L.info(f"generate_brain_surfaces returning {len(final_result)} meshes: {list(final_result.keys())}")
    if len(result) != len(final_result): L.warning(f"{len(result) - len(final_result)} meshes excluded.")
    return final_result

# --- 5ttgen Function ---
# ... (As previously corrected) ...
def run_5ttgen_hsvs_save_temp_bids( subject_id: str, fs_subject_dir: str, subject_work_dir: str, session_id: Optional[str] = None, nocrop: bool = True, sgm_amyg_hipp: bool = True, verbose: bool = False ) -> Optional[str]:
    subject_work_dir_path = Path(subject_work_dir); sub_label = f"sub-{subject_id}"; ses_label = f"ses-{session_id}" if session_id else None; persistent_5ttgen_path = subject_work_dir_path / "5ttgen_persistent_work"
    try: persistent_5ttgen_path.mkdir(parents=True, exist_ok=True); L.info(f"Ensured 5ttgen persistent dir: {persistent_5ttgen_path}")
    except Exception as e: L.error(f"Failed create 5ttgen dir {persistent_5ttgen_path}: {e}"); return None
    fname_5tt_parts = [sub_label];
    if ses_label: fname_5tt_parts.append(ses_label);
    fname_5tt_parts.append("desc-5ttgen_dseg.nii.gz"); final_5tt_out = persistent_5ttgen_path / "_".join(fname_5tt_parts)
    cmd = ["5ttgen", "hsvs", str(Path(fs_subject_dir)), str(final_5tt_out), "-scratch", str(persistent_5ttgen_path)];
    if nocrop: cmd.append("-nocrop");
    if sgm_amyg_hipp: cmd.append("-sgm_amyg_hipp");
    cmd.append("-nocleanup"); L.info(f"Running 5ttgen: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True); L.info("5ttgen completed."); L.debug(f"stdout: {proc.stdout}"); L.debug(f"stderr: {proc.stderr}")
        if not final_5tt_out.exists(): L.warning(f"Main 5ttgen NIfTI {final_5tt_out} not found.")
        nested_dirs = list(persistent_5ttgen_path.glob("5ttgen-tmp-*"))
        if not nested_dirs: L.error(f"No '5ttgen-tmp-*' dir in {persistent_5ttgen_path}. VTK missing?")
        else: L.info(f"Found 5ttgen tmp dir: {nested_dirs[0]}.")
        L.info(f"5ttgen outputs retained at: {persistent_5ttgen_path}"); return str(persistent_5ttgen_path)
    except subprocess.CalledProcessError as e: L.error(f"5ttgen failed (Code {e.returncode}).", exc_info=verbose); L.error(f"Stderr: {e.stderr}")
    except Exception as e: L.error(f"Unexpected error during 5ttgen: {e}", exc_info=verbose)
    L.warning(f"5ttgen failed. Dir {persistent_5ttgen_path} may have partial outputs."); return None

# --- VTK Loading Function ---
# ... (As previously corrected, depends on VTK_AVAILABLE) ...
def load_subcortical_and_ventricle_meshes( five_ttgen_persistent_dir_str: str ) -> Dict[str, trimesh.Trimesh]:
    loaded_meshes: Dict[str, trimesh.Trimesh] = {};
    if not VTK_AVAILABLE: L.error("VTK unavailable."); return loaded_meshes
    persistent_work_dir = Path(five_ttgen_persistent_dir_str);
    if not persistent_work_dir.is_dir(): L.error(f"5ttgen dir not found: {persistent_work_dir}"); return loaded_meshes
    nested_dirs = list(persistent_work_dir.glob("5ttgen-tmp-*"));
    if not nested_dirs: L.warning(f"No '5ttgen-tmp-*' dir in {persistent_work_dir}. Searching root."); search_dir = persistent_work_dir
    else: search_dir = nested_dirs[0]; L.info(f"Searching VTK in: {search_dir}")
    skipped = 0; sgm_pattern = str(search_dir / "first-*_transformed.vtk"); sgm_files = glob.glob(sgm_pattern); L.info(f"Found {len(sgm_files)} SGM files.")
    for p_str in sgm_files:
        p = Path(p_str); fn = p.name;
        if fn.startswith("first-") and fn.endswith("_transformed.vtk"): name = fn[len("first-"):-len("_transformed.vtk")]; key = f"subcortical-{name}"
        else: L.warning(f"Skipping SGM file: {fn}"); skipped+=1; continue
        if not name: L.warning(f"Skipping SGM empty name: {fn}"); skipped+=1; continue
        try: poly = _read_vtk_polydata(str(p), L); mesh = _vtk_polydata_to_trimesh(poly) if poly else None
        except Exception as e: L.warning(f"Err proc SGM {fn} ({key}): {e}"); skipped+=1; continue
        if mesh and not mesh.is_empty: loaded_meshes[key] = mesh; L.info(f"Loaded SGM: {key}")
        else: L.debug(f"Skip empty/failed SGM {key}"); skipped+=1
    other_files = [p for p in search_dir.glob("*.vtk") if not p.name.startswith("first-")]; L.info(f"Found {len(other_files)} other VTK files.")
    vent_kw = ["ventricle", "vent", "choroid", "plexus", "latvent", "inf-lat-vent"]; vessel_kw = ["vessel"]; csf_kw = ["csf"]; skip_kw = ["_init", "gm_", "wm_", "brain_"]
    for p in other_files:
        fn_low = p.name.lower(); base = p.stem;
        if any(sk in fn_low for sk in skip_kw): L.debug(f"Skip intermediate VTK: {p.name}"); continue
        prefix = ""; name_part = base.replace("_transformed", "").replace("_surf", "").replace("_vol", "").replace("seg_", "")
        if any(kw in fn_low for kw in vent_kw): prefix = "ventricle"
        elif any(kw in fn_low for kw in vessel_kw): prefix = "vessel"
        elif any(kw in fn_low for kw in csf_kw): prefix = "csf"
        else: L.debug(f"Skip unrecog VTK: {p.name}"); skipped+=1; continue
        key = f"{prefix}-{name_part}"
        if key in loaded_meshes: L.debug(f"Key {key} exists. Skip {p.name}."); continue
        try: poly = _read_vtk_polydata(str(p), L); mesh = _vtk_polydata_to_trimesh(poly) if poly else None
        except Exception as e: L.warning(f"Err proc other VTK {p.name} ({key}): {e}"); skipped+=1; continue
        if mesh and not mesh.is_empty: loaded_meshes[key] = mesh; L.info(f"Loaded {prefix}: {key}")
        else: L.debug(f"Skip empty/failed {prefix} {key}"); skipped+=1
    if skipped > 0: L.warning(f"Skipped {skipped} VTK files.")
    L.info(f"VTK loading done. {len(loaded_meshes)} meshes loaded: {list(loaded_meshes.keys())}"); return loaded_meshes

# --- VTK Helper Functions ---
if VTK_AVAILABLE:
    def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[vtk.vtkPolyData]:
        logger.debug(f"Reading VTK: {path}")
        if not Path(path).exists(): logger.error(f"VTK file not found: {path}"); return None
        reader_types = [vtk.vtkSTLReader, vtk.vtkPolyDataReader, vtk.vtkXMLPolyDataReader, vtk.vtkGenericDataObjectReader]
        for reader_class in reader_types:
            try:
                reader = reader_class(); reader.SetFileName(path); reader.Update()
                poly_data_output = None
                if isinstance(reader, vtk.vtkGenericDataObjectReader):
                    if reader.IsFilePolyData(): poly_data_output = reader.GetPolyDataOutput()
                    else: logger.debug(f"vtkGeneric reports {path} not PolyData."); continue
                elif hasattr(reader, 'GetOutput') and isinstance(reader.GetOutput(), vtk.vtkPolyData): poly_data_output = reader.GetOutput()
                if poly_data_output and poly_data_output.GetNumberOfPoints() > 0: logger.info(f"Read {Path(path).name} via {reader_class.__name__}."); return poly_data_output
                else: logger.debug(f"{reader_class.__name__} gave no/empty PolyData for {path}.")
            except Exception as e_reader: logger.debug(f"{reader_class.__name__} failed for {path}: {e_reader}")
        logger.warning(f"Could not read valid PolyData from VTK file {path}."); return None

    def _vtk_polydata_to_trimesh(poly_data: Optional[vtk.vtkPolyData]) -> Optional[trimesh.Trimesh]:
        if not VTK_AVAILABLE or poly_data is None or poly_data.GetNumberOfPoints() == 0: return None
        num_pts = poly_data.GetPoints().GetNumberOfPoints(); verts = np.zeros((num_pts, 3))
        for i in range(num_pts): verts[i, :] = poly_data.GetPoints().GetPoint(i)
        faces = []; ids = vtk.vtkIdList()
        polys = poly_data.GetPolys();
        if polys and polys.GetNumberOfCells() > 0:
            polys.InitTraversal();
            while polys.GetNextCell(ids): faces.append([ids.GetId(j) for j in range(ids.GetNumberOfIds())])
        elif poly_data.GetStrips() and poly_data.GetStrips().GetNumberOfCells() > 0:
             strips = poly_data.GetStrips(); strips.InitTraversal(); L.warning("VTK has strips, may be lossy.")
             while strips.GetNextCell(ids):
                 for j in range(ids.GetNumberOfIds() - 2): faces.append([ids.GetId(j + (j % 2)), ids.GetId(j + 1 - (j % 2)), ids.GetId(j + 2)])
        if not faces: L.warning("VTK has points but no faces."); return trimesh.Trimesh(vertices=verts)
        try: mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True); return None if mesh.is_empty else mesh
        except Exception as e: L.error(f"Failed Trimesh creation: {e}"); return None

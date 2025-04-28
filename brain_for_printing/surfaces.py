# brain_for_printing/surfaces.py

"""
surfaces.py
-----------
Generate multiple brain surfaces (cortical + optional brainstem, cerebellum, etc.)
in T1 or MNI space based on FreeSurfer aseg labels. Includes function to run
5ttgen hsvs and save its working directory to BIDS derivatives.
"""
import os
import uuid
import shutil
import trimesh
import logging
import subprocess
import tempfile
import json
from pathlib import Path
# Corrected typing imports for older Python versions
from typing import List, Tuple, Dict, Optional, Union

# Assuming these local imports are correct relative to surfaces.py
from .io_utils import run_cmd, flexible_match, first_match
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .warp_utils import generate_mrtrix_style_warp, warp_gifti_vertices
from . import constants as const


# Configure logger for this module
L = logging.getLogger(__name__)

# --- Helper Function for T1 Mask Extraction ---

def _extract_structure_mask_t1(
    aseg_t1_nii: str,
    label_ids: List[int],
    output_mask_nii: str,
    tmp_dir: str = ".",
    verbose: bool = False
) -> bool:
    """
    Helper to binarize a T1-space aseg NIfTI based on label IDs.
    """
    # ... (function body remains the same) ...
    if not Path(aseg_t1_nii).exists():
        L.error(f"Input aseg file not found: {aseg_t1_nii}")
        return False

    match_str = [str(lbl) for lbl in label_ids]
    L.info(f"Extracting labels {match_str} from {Path(aseg_t1_nii).name}")

    try:
        run_cmd([
            "mri_binarize",
            "--i", aseg_t1_nii,
            "--match", *match_str,
            "--o", output_mask_nii
        ], verbose=verbose)
    except Exception as e:
        L.error(f"Failed to binarize {aseg_t1_nii} for labels {match_str}: {e}")
        if Path(output_mask_nii).exists():
            os.remove(output_mask_nii)
        return False

    if not Path(output_mask_nii).exists() or Path(output_mask_nii).stat().st_size == 0:
        L.error(f"Binary mask {output_mask_nii} was not created or is empty.")
        return False

    return True


# --- Main Structure Extraction Function (T1 and MNI) ---

def extract_structure_surface(
    subjects_dir: str,
    subject_id: str,
    label_ids: List[int],
    output_tag: str,
    space: str = "T1",
    tmp_dir: str = ".",
    verbose: bool = False,
    session: Optional[str] = None, # Use Optional[str] instead of str | None
    run: Optional[str] = None     # Use Optional[str] instead of str | None
) -> Optional[str]:              # Use Optional[str] instead of str | None
    """
    Extracts a surface mesh for a structure defined by label IDs in T1 or MNI space.
    """
    # ... (function body remains the same) ...
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    anat_dir = Path(subjects_dir) / subject_id / "anat"
    output_mask_nii = Path(tmp_dir) / f"{output_tag}_mask_{space}.nii.gz"
    output_gii = Path(tmp_dir) / f"{output_tag}_{space}.surf.gii"

    aseg_in_target_space: Optional[str] = None

    try:
        if space.upper() == "T1":
            aseg_in_target_space = flexible_match(
                base_dir=anat_dir, subject_id=subject_id, descriptor="desc-aseg",
                suffix="dseg", ext=".nii.gz", session=session, run=run
            )
        elif space.upper() == "MNI":
            aseg_t1_path = flexible_match(
                base_dir=anat_dir, subject_id=subject_id, descriptor="desc-aseg",
                suffix="dseg", ext=".nii.gz", session=session, run=run
            )
            xfm_t1_to_mni = flexible_match(
                base_dir=anat_dir, subject_id=subject_id,
                descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image", suffix="xfm",
                ext=".h5", session=session, run=run
            )
            mni_template = flexible_match(
                base_dir=anat_dir, subject_id=subject_id,
                descriptor="space-MNI152NLin2009cAsym_res-", suffix="T1w",
                ext=".nii.gz", session=session, run=run
            )
            aseg_in_target_space_path = Path(tmp_dir) / f"aseg_in_mni_{uuid.uuid4().hex[:4]}.nii.gz"
            L.info(f"Warping aseg to MNI space: {aseg_in_target_space_path.name}")
            run_cmd([
                "antsApplyTransforms", "-d", "3", "-i", aseg_t1_path,
                "-o", str(aseg_in_target_space_path), "-r", mni_template,
                "-t", xfm_t1_to_mni, "-n", "NearestNeighbor"
            ], verbose=verbose)
            aseg_in_target_space = str(aseg_in_target_space_path)
        else:
            L.error(f"Unsupported space: {space}")
            return None
    except FileNotFoundError as e:
        L.error(f"Could not find required file for space '{space}': {e}")
        return None
    except Exception as e:
        L.error(f"Error finding/warping aseg for space '{space}': {e}")
        return None

    if not aseg_in_target_space or not Path(aseg_in_target_space).exists():
        L.error("Failed to obtain aseg segmentation in target space.")
        return None

    success = _extract_structure_mask_t1(
        aseg_t1_nii=aseg_in_target_space, label_ids=label_ids,
        output_mask_nii=str(output_mask_nii), tmp_dir=tmp_dir, verbose=verbose
    )

    if not success:
        L.error(f"Failed to create binary mask for {output_tag} in {space} space.")
        return None

    try:
        volume_to_gifti(str(output_mask_nii), str(output_gii), level=0.5)
        L.info(f"Successfully created surface: {output_gii.name}")
        return str(output_gii)
    except Exception as e:
        L.error(f"Failed to convert mask to surface for {output_tag}: {e}")
        if Path(output_gii).exists():
            os.remove(output_gii)
        return None


# --- Main Surface Generation Function ---

def generate_brain_surfaces(
    subjects_dir: str,
    subject_id: str,
    space: str = "T1",
    surfaces: Tuple[str, ...] = ("pial",),
    extract_structures: Optional[List[str]] = None,
    no_fill_structures: Optional[List[str]] = None,
    no_smooth_structures: Optional[List[str]] = None,
    out_warp: str = "warp.nii",
    run: Optional[str] = None,
    session: Optional[str] = None,
    verbose: bool = False,
    tmp_dir: Optional[str] = None
) -> Dict[str, Optional[trimesh.Trimesh]]: # Use Dict and Optional
    """
    Generates cortical surfaces AND extracts other specified structures (brainstem, etc.)
    in T1 or MNI space based on FreeSurfer outputs.
    """
    # ... (function body remains the same, type hints within are usually fine) ...
    if extract_structures is None: extract_structures = []
    if no_fill_structures is None: no_fill_structures = []
    if no_smooth_structures is None: no_smooth_structures = []

    STRUCTURE_LABEL_MAP = {
        "brainstem": const.BRAINSTEM_LABEL,
        "cerebellum_wm": const.CEREBELLUM_WM_LABELS,
        "cerebellum_cortex": const.CEREBELLUM_CORTEX_LABELS,
        "cerebellum": const.CEREBELLUM_LABELS,
        "corpus_callosum": const.CORPUS_CALLOSUM_LABELS,
    }

    valid_extract_structures = []
    for struct_name in extract_structures:
        if struct_name in STRUCTURE_LABEL_MAP:
            valid_extract_structures.append(struct_name)
        else:
            L.warning(f"Requested structure '{struct_name}' not recognized or constants not defined. Skipping.")
    extract_structures = valid_extract_structures

    SURF_NAME_MAP = {
        "pial": "pial", "mid": "midthickness", "white": "smoothwm"
    }
    def resolve_suffix(surf_type):
        return SURF_NAME_MAP.get(surf_type, surf_type)

    local_tmp = False
    temp_dir_obj = None # Define for potential cleanup in except block
    if not tmp_dir:
        try:
             temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"_tmp_surf_{subject_id}_")
             tmp_dir = temp_dir_obj.name
             local_tmp = True
             if verbose: L.info(f"Created local temp dir => {tmp_dir}")
        except Exception as e:
            L.error(f"Failed to create temporary directory: {e}")
            return {}
    else:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    anat_dir = Path(subjects_dir) / subject_id / "anat"
    result: Dict[str, Optional[trimesh.Trimesh]] = {} # Use Dict and Optional
    for surf_type in surfaces:
        result[f"{surf_type}_L"] = None
        result[f"{surf_type}_R"] = None
    for struct_name in extract_structures:
         result[struct_name] = None

    t1_gifti_paths = {}
    warp_field = None

    try:
        for surf_type in surfaces:
            actual_name = resolve_suffix(surf_type)
            lh_file = flexible_match(
                base_dir=anat_dir, subject_id=subject_id, descriptor=None,
                suffix=f"{actual_name}.surf", hemi="hemi-L", ext=".gii", run=run, session=session
            )
            rh_file = flexible_match(
                base_dir=anat_dir, subject_id=subject_id, descriptor=None,
                suffix=f"{actual_name}.surf", hemi="hemi-R", ext=".gii", run=run, session=session
            )
            t1_gifti_paths[f"{surf_type}_L"] = lh_file
            t1_gifti_paths[f"{surf_type}_R"] = rh_file

        if space.upper() == "MNI":
            mni_template = flexible_match(
                base_dir=anat_dir, subject_id=subject_id, descriptor="space-MNI152NLin2009cAsym_res-",
                suffix="T1w", ext=".nii.gz", session=session, run=run
            )
            t1_preproc = flexible_match(
                base_dir=anat_dir, subject_id=subject_id, descriptor="desc-preproc",
                suffix="T1w", ext=".nii.gz", session=session, run=run
            )
            xfm_mni_to_t1 = flexible_match(
                base_dir=anat_dir, subject_id=subject_id,
                descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image",
                suffix="xfm", ext=".h5", session=session, run=run
            )
            if verbose:
                L.info(f"MNI template matched => {mni_template}")
                L.info(f"T1 preproc matched   => {t1_preproc}")
                L.info(f"Transform (MNI->T1)  => {xfm_mni_to_t1}")

            warp_field_path = Path(tmp_dir) / out_warp
            generate_mrtrix_style_warp(
                 mni_template, t1_preproc, xfm_mni_to_t1,
                 out_warp, tmp_dir, verbose
            )
            warp_field = str(warp_field_path)

            for surf_type in surfaces:
                lh_out = Path(tmp_dir) / f"L_{surf_type}_mni.gii"
                rh_out = Path(tmp_dir) / f"R_{surf_type}_mni.gii"
                warp_gifti_vertices(t1_gifti_paths[f"{surf_type}_L"], warp_field, str(lh_out), verbose=verbose)
                warp_gifti_vertices(t1_gifti_paths[f"{surf_type}_R"], warp_field, str(rh_out), verbose=verbose)
                result[f"{surf_type}_L"] = gifti_to_trimesh(str(lh_out))
                result[f"{surf_type}_R"] = gifti_to_trimesh(str(rh_out))
        else:
            for surf_type in surfaces:
                result[f"{surf_type}_L"] = gifti_to_trimesh(t1_gifti_paths[f"{surf_type}_L"])
                result[f"{surf_type}_R"] = gifti_to_trimesh(t1_gifti_paths[f"{surf_type}_R"])
    except FileNotFoundError as e:
         L.error(f"Failed to find required cortical surface or transform file: {e}")
         if local_tmp and temp_dir_obj: temp_dir_obj.cleanup()
         return result
    except Exception as e:
         L.error(f"Error processing cortical surfaces: {e}")
         if local_tmp and temp_dir_obj: temp_dir_obj.cleanup()
         return result

    for struct_name in extract_structures:
        label_ids = STRUCTURE_LABEL_MAP[struct_name]
        L.info(f"Extracting structure: {struct_name} in {space} space...")
        struct_gii_path = extract_structure_surface(
            subjects_dir=subjects_dir, subject_id=subject_id, label_ids=label_ids,
            output_tag=struct_name, space=space, tmp_dir=tmp_dir,
            verbose=verbose, session=session, run=run
        )
        if struct_gii_path and Path(struct_gii_path).exists():
            try:
                mesh = gifti_to_trimesh(struct_gii_path)
                do_fill = struct_name not in no_fill_structures
                do_smooth = struct_name not in no_smooth_structures
                if do_fill:
                    try: trimesh.repair.fill_holes(mesh); L.info(f"Filled holes for {struct_name}")
                    except Exception as e_fill: L.warning(f"Hole filling failed for {struct_name}: {e_fill}")
                if do_smooth:
                    try: trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10); L.info(f"Smoothed mesh for {struct_name}")
                    except Exception as e_smooth: L.warning(f"Smoothing failed for {struct_name}: {e_smooth}")
                mesh.fix_normals()
                result[struct_name] = mesh
            except Exception as e_mesh:
                 L.warning(f"Failed to load or process mesh for {struct_name} from {struct_gii_path}: {e_mesh}")
                 result[struct_name] = None
        else:
            L.warning(f"Failed to generate surface file for {struct_name}")
            result[struct_name] = None

    if local_tmp and temp_dir_obj:
        try:
            temp_dir_obj.cleanup()
            if verbose: L.info(f"Removed local temp dir => {tmp_dir}")
        except Exception as e_clean:
             L.warning(f"Failed to remove temporary directory {tmp_dir}: {e_clean}")

    return result


# --- 5ttgen Function (kept here as previously placed) ---

def run_5ttgen_hsvs_save_temp_bids(
    subject_id: str,
    t1w_image: str,
    bids_root_dir: str,
    pipeline_name: str = "brain_for_printing",
    session_id: Optional[str] = None, # Corrected type hint
    # Add flags for 5ttgen if needed
    premasked: bool = False,
    nocrop: bool = True,
    sgm_amyg_hipp: bool = True,
) -> Optional[str]: # Corrected type hint
    """
    Runs 5ttgen hsvs, saves its *entire temporary working directory* to a
    BIDS derivatives structure, and returns the path to this saved directory.
    """
    # ... (function body remains the same, but uses L defined above) ...
    sub_label = f"sub-{subject_id}"
    ses_label = f"ses-{session_id}" if session_id else None

    deriv_pipeline_dir_parts = [bids_root_dir, "derivatives", pipeline_name]
    deriv_subject_dir_parts = deriv_pipeline_dir_parts + [sub_label]
    if ses_label:
        deriv_subject_dir_parts.append(ses_label)
    deriv_anat_dir = Path(os.path.join(*deriv_subject_dir_parts)) / "anat"

    dir_name_parts = [sub_label]
    if ses_label:
        dir_name_parts.append(ses_label)
    dir_name_parts.append("desc-5ttgenwork_directory")
    final_bids_temp_dir_path = deriv_anat_dir / "_".join(dir_name_parts)

    deriv_anat_dir.mkdir(parents=True, exist_ok=True)

    if final_bids_temp_dir_path.exists():
        L.error(f"Target BIDS directory already exists: {final_bids_temp_dir_path}")
        L.error("Please remove it or choose a different pipeline/descriptor name.")
        return None

    initial_temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"{sub_label}_5ttgen_initial_")
    initial_temp_dir = initial_temp_dir_obj.name
    L.info(f"Created initial temporary directory for 5ttgen run: {initial_temp_dir}")

    temp_output_5tt = Path(initial_temp_dir) / "5tt_output.nii.gz"

    cmd = ["5ttgen", "hsvs", t1w_image, str(temp_output_5tt)]
    cmd.extend(["-tempdir", initial_temp_dir])
    if premasked: cmd.append("-premasked")
    if nocrop: cmd.append("-nocrop")
    if sgm_amyg_hipp: cmd.append("-sgm_amyg_hipp")

    L.info(f"Running 5ttgen command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        L.info("5ttgen hsvs completed successfully.")

        L.info(f"Moving {initial_temp_dir} to {final_bids_temp_dir_path}")
        if final_bids_temp_dir_path.exists():
             raise FileExistsError(f"Target directory {final_bids_temp_dir_path} appeared unexpectedly.")
        shutil.move(initial_temp_dir, str(final_bids_temp_dir_path))
        L.info(f"Successfully moved 5ttgen working directory to BIDS derivatives.")

    except subprocess.CalledProcessError as e:
        L.error(f"5ttgen hsvs failed with return code {e.returncode}")
        L.error(f"Command: {' '.join(e.cmd)}")
        L.error(f"STDERR:\n{e.stderr}")
        L.error(f"STDOUT:\n{e.stdout}")
        initial_temp_dir_obj.cleanup()
        L.info(f"Removed initial temporary directory due to failure: {initial_temp_dir}")
        return None
    except Exception as e:
        L.error(f"An unexpected error occurred during or after 5ttgen: {e}")
        initial_temp_dir_obj.cleanup()
        L.info(f"Removed initial temporary directory due to error: {initial_temp_dir}")
        return None

    pipeline_dataset_desc = Path(bids_root_dir) / "derivatives" / pipeline_name / "dataset_description.json"
    if not pipeline_dataset_desc.exists():
        L.info(f"Creating minimal dataset_description.json for derivative pipeline '{pipeline_name}'")
        pipeline_dataset_desc.parent.mkdir(parents=True, exist_ok=True)
        desc_content = {
            "Name": f"{pipeline_name} Derivatives",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": pipeline_name}]
        }
        try:
            with open(pipeline_dataset_desc, 'w') as f:
                json.dump(desc_content, f, indent=2)
        except Exception as e:
            L.warning(f"Could not write dataset_description.json: {e}")

    return str(final_bids_temp_dir_path)

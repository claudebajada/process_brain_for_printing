# brain_for_printing/surfaces.py

"""
surfaces.py
-----------
Generate multiple brain surfaces (cortical + optional brainstem, cerebellum, etc.)
in T1 or MNI space based on FreeSurfer aseg labels. Includes function to run
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
import glob # Added for globbing in load_subcortical_and_ventricle_meshes
from pathlib import Path
# Corrected typing imports for older Python versions
from typing import List, Tuple, Dict, Optional, Union

# Assuming these local imports are correct relative to surfaces.py
from .io_utils import run_cmd, flexible_match, first_match
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .warp_utils import generate_mrtrix_style_warp, warp_gifti_vertices
from . import constants as const


# Configure logger for this module
L = logging.getLogger(__name__) # Define logger once for the module

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

    Args:
        aseg_t1_nii (str): Path to the input aseg NIfTI file in T1 space.
        label_ids (List[int]): List of integer label IDs to match.
        output_mask_nii (str): Path for the output binary mask NIfTI file.
        tmp_dir (str): Temporary directory for intermediate files.
        verbose (bool): Verbosity flag.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not Path(aseg_t1_nii).exists():
        L.error(f"Input aseg file not found: {aseg_t1_nii}")
        return False

    match_str = [str(lbl) for lbl in label_ids]
    L.info(f"Extracting labels {match_str} from {Path(aseg_t1_nii).name}")

    try:
        # Direct binarization using mri_binarize
        run_cmd([
            "mri_binarize",
            "--i", aseg_t1_nii,
            "--match", *match_str,  # Unpack the list of strings
            "--o", output_mask_nii
        ], verbose=verbose)
        # Optional: Ensure pure binary mask if needed
        # run_cmd(["fslmaths", output_mask_nii, "-bin", output_mask_nii], verbose=verbose)

    except Exception as e:
        L.error(f"Failed to binarize {aseg_t1_nii} for labels {match_str}: {e}")
        # Clean up potentially incomplete output file
        if Path(output_mask_nii).exists():
            try:
                os.remove(output_mask_nii)
            except OSError:
                pass # Ignore if removal fails
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
    output_tag: str, # e.g., "brainstem", "cerebellum_wm"
    space: str = "T1", # "T1" or "MNI"
    tmp_dir: str = ".",
    verbose: bool = False,
    session: Optional[str] = None, # Use Optional[str] instead of str | None
    run: Optional[str] = None     # Use Optional[str] instead of str | None
) -> Optional[str]:              # Use Optional[str] instead of str | None
    """
    Extracts a surface mesh for a structure defined by label IDs in T1 or MNI space.

    Args:
        subjects_dir (str): Path to derivatives (e.g., fMRIPrep output) or FreeSurfer SUBJECTS_DIR.
        subject_id (str): Subject identifier (e.g., "sub-01").
        label_ids (List[int]): List of integer label IDs defining the structure.
        output_tag (str): A descriptive tag for filenames (e.g., "brainstem").
        space (str): Space for extraction ("T1" or "MNI"). Default is "T1".
        tmp_dir (str): Path to the temporary working directory.
        verbose (bool): Enable verbose output.
        session (Optional[str]): BIDS session ID.
        run (Optional[str]): BIDS run ID.

    Returns:
        Optional[str]: Path to the generated GIFTI surface file, or None on failure.
    """
    # Ensure tmp_dir exists
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # Clean subject_id prefix if present
    subject_id_clean = subject_id.replace('sub-', '')
    anat_dir = Path(subjects_dir) / f"sub-{subject_id_clean}" / "anat" # Construct path correctly
    output_mask_nii = Path(tmp_dir) / f"{output_tag}_mask_{space}_{uuid.uuid4().hex[:4]}.nii.gz" # Add uuid
    output_gii = Path(tmp_dir) / f"{output_tag}_{space}.surf.gii"

    # --- Step 1: Get the aseg segmentation in the correct space ---
    aseg_in_target_space: Optional[str] = None

    try:
        if space.upper() == "T1":
            aseg_in_target_space = flexible_match(
                base_dir=anat_dir,
                subject_id=f"sub-{subject_id_clean}",
                descriptor="desc-aseg",
                suffix="dseg",
                ext=".nii.gz",
                session=session,
                run=run
            )

        elif space.upper() == "MNI":
            # Warp aseg from T1 to MNI
            aseg_t1_path = flexible_match(
                base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}", descriptor="desc-aseg",
                suffix="dseg", ext=".nii.gz", session=session, run=run
            )
            # Find necessary transforms and template (robustly)
            xfm_t1_to_mni = flexible_match(
                base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}",
                descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image", suffix="xfm",
                ext=".h5", session=session, run=run
            )
            mni_template = flexible_match(
                base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}",
                descriptor="space-MNI152NLin2009cAsym_res-", # Allow diff resolutions
                suffix="T1w", ext=".nii.gz", session=session, run=run # Find appropriate template
            )

            aseg_in_target_space_path = Path(tmp_dir) / f"aseg_in_mni_{uuid.uuid4().hex[:4]}.nii.gz"
            L.info(f"Warping aseg to MNI space: {aseg_in_target_space_path.name}")
            run_cmd([
                "antsApplyTransforms", "-d", "3",
                "-i", aseg_t1_path,
                "-o", str(aseg_in_target_space_path),
                "-r", mni_template,
                "-t", xfm_t1_to_mni,
                "-n", "NearestNeighbor" # Use Nearest Neighbor for label maps
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

    # --- Step 2: Extract the binary mask for the structure ---
    success = _extract_structure_mask_t1(
        aseg_t1_nii=aseg_in_target_space, # Use the (potentially warped) aseg
        label_ids=label_ids,
        output_mask_nii=str(output_mask_nii),
        tmp_dir=tmp_dir,
        verbose=verbose
    )

    if not success:
        L.error(f"Failed to create binary mask for {output_tag} in {space} space.")
        return None

    # --- Step 3: Convert binary mask volume to GIFTI surface ---
    try:
        volume_to_gifti(str(output_mask_nii), str(output_gii), level=0.5)
        L.info(f"Successfully created surface: {output_gii.name}")
        return str(output_gii)
    except Exception as e:
        L.error(f"Failed to convert mask to surface for {output_tag}: {e}")
        if Path(output_gii).exists(): # Clean up potentially incomplete output
             try: os.remove(output_gii)
             except OSError: pass
        return None


# --- Main Surface Generation Function ---

def generate_brain_surfaces(
    subjects_dir: str,
    subject_id: str,
    space: str = "T1",
    surfaces: Tuple[str, ...] = ("pial",), # e.g., ("pial", "white", "inflated")
    extract_structures: Optional[List[str]] = None,
    no_fill_structures: Optional[List[str]] = None,
    no_smooth_structures: Optional[List[str]] = None,
    out_warp: str = "warp.nii",
    run: Optional[str] = None,
    session: Optional[str] = None,
    verbose: bool = False,
    tmp_dir: Optional[str] = None
) -> Dict[str, Optional[trimesh.Trimesh]]:
    """
    Generates cortical surfaces (including inflated) AND extracts other specified structures
    in T1 or MNI space based on FreeSurfer outputs.

    Args:
        subjects_dir (str): Path to derivatives or subject data (with /anat subfolder).
        subject_id (str): Subject identifier (e.g. "sub-01").
        space (str): "T1" (native) or "MNI" (warped).
        surfaces (Tuple[str,...]): Which cortical surfaces to load (e.g. ("pial","white","inflated")).
        extract_structures (Optional[List[str]]): List of additional structures to extract
            by name (e.g., "brainstem", "cerebellum_wm").
        no_fill_structures (Optional[List[str]]): List of extracted structure names
            to skip hole-filling on.
        no_smooth_structures (Optional[List[str]]): List of extracted structure names
            to skip smoothing on.
        out_warp (str): 4D warp field filename if warping to MNI.
        run (Optional[str]): BIDS run ID.
        session (Optional[str]): BIDS session ID.
        verbose (bool): Enable verbose output.
        tmp_dir (Optional[str]): If provided, use this folder for intermediate files;
                           otherwise a new one is created and removed.

    Returns:
        Dict[str, Optional[trimesh.Trimesh]]: Trimesh objects keyed by surface/structure name.

    Raises:
        ValueError: If 'inflated' surface is requested in 'MNI' space.
    """
    # Initialize structure lists if None
    if extract_structures is None: extract_structures = []
    if no_fill_structures is None: no_fill_structures = []
    if no_smooth_structures is None: no_smooth_structures = []

    # --- Map structure names to label lists ---
    STRUCTURE_LABEL_MAP = {
        "brainstem": const.BRAINSTEM_LABEL,
        "cerebellum_wm": const.CEREBELLUM_WM_LABELS,
        "cerebellum_cortex": const.CEREBELLUM_CORTEX_LABELS,
        "cerebellum": const.CEREBELLUM_LABELS,
        "corpus_callosum": const.CORPUS_CALLOSUM_LABELS,
    }
    # Validate requested structures
    valid_extract_structures = [s for s in extract_structures if s in STRUCTURE_LABEL_MAP]
    if len(valid_extract_structures) != len(extract_structures):
         invalid_structs = set(extract_structures) - set(valid_extract_structures)
         L.warning(f"Requested structure(s) not recognized or constants not defined, skipping: {invalid_structs}")
    extract_structures = valid_extract_structures # Use only valid ones

    # Map user-requested cortical types to BIDS suffixes, including inflated
    SURF_NAME_MAP = {
        "pial": "pial", "mid": "midthickness", "white": "smoothwm", "inflated": "inflated"
    }
    processed_surf_types = set() # Keep track of valid requested types
    for s_type in surfaces:
        if s_type in SURF_NAME_MAP:
             processed_surf_types.add(s_type)
        else:
             L.warning(f"Unrecognized surface type '{s_type}' requested. Skipping.")

    # --- Check for invalid inflated + MNI combination ---
    if "inflated" in processed_surf_types and space.upper() == "MNI":
        # Raise error here to be caught by the CLI
        raise ValueError("Inflated surfaces cannot be generated in MNI space. Please use --space T1 for inflated surfaces.")

    # --- Setup Temp Dir ---
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
            return {} # Return empty if temp dir fails
    else:
        # Ensure provided tmp_dir exists if given
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # Clean subject_id prefix if present
    subject_id_clean = subject_id.replace('sub-', '')
    anat_dir = Path(subjects_dir) / f"sub-{subject_id_clean}" / "anat" # Construct path correctly

    # --- Initialize Result Dictionary ---
    result: Dict[str, Optional[trimesh.Trimesh]] = {}
    for surf_type in processed_surf_types:
        result[f"{surf_type}_L"] = None
        result[f"{surf_type}_R"] = None
    for struct_name in extract_structures:
        result[struct_name] = None

    # --- Cortical Surface Handling ---
    t1_gifti_paths = {}
    warp_field = None

    try:
        # Find T1 paths for all requested cortical types
        for surf_type in processed_surf_types:
            actual_name = SURF_NAME_MAP[surf_type] # Already validated
            lh_file = flexible_match(
                base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}", descriptor=None,
                suffix=f"{actual_name}.surf", hemi="hemi-L", ext=".gii", run=run, session=session
            )
            rh_file = flexible_match(
                base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}", descriptor=None,
                suffix=f"{actual_name}.surf", hemi="hemi-R", ext=".gii", run=run, session=session
            )
            t1_gifti_paths[f"{surf_type}_L"] = lh_file
            t1_gifti_paths[f"{surf_type}_R"] = rh_file

        # --- MNI Space Warping (Inflated already excluded) ---
        if space.upper() == "MNI":
            mni_template = flexible_match(base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}", descriptor="space-MNI152NLin2009cAsym_res-", suffix="T1w", ext=".nii.gz", session=session, run=run)
            t1_preproc = flexible_match(base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}", descriptor="desc-preproc", suffix="T1w", ext=".nii.gz", session=session, run=run)
            xfm_mni_to_t1 = flexible_match(base_dir=anat_dir, subject_id=f"sub-{subject_id_clean}", descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image", suffix="xfm", ext=".h5", session=session, run=run)
            if verbose: L.info(f"MNI->T1 Transform: {xfm_mni_to_t1}")
            warp_field_path = Path(tmp_dir) / out_warp
            generate_mrtrix_style_warp(mni_template, t1_preproc, xfm_mni_to_t1, out_warp, tmp_dir, verbose)
            warp_field = str(warp_field_path)

            for surf_type in processed_surf_types:
                lh_out = Path(tmp_dir) / f"L_{surf_type}_mni.gii"; rh_out = Path(tmp_dir) / f"R_{surf_type}_mni.gii"
                warp_gifti_vertices(t1_gifti_paths[f"{surf_type}_L"], warp_field, str(lh_out), verbose=verbose)
                warp_gifti_vertices(t1_gifti_paths[f"{surf_type}_R"], warp_field, str(rh_out), verbose=verbose)
                try: result[f"{surf_type}_L"] = gifti_to_trimesh(str(lh_out))
                except Exception as e: L.warning(f"Failed to load MNI mesh {lh_out.name}: {e}")
                try: result[f"{surf_type}_R"] = gifti_to_trimesh(str(rh_out))
                except Exception as e: L.warning(f"Failed to load MNI mesh {rh_out.name}: {e}")

        else: # T1 space
            for surf_type in processed_surf_types:
                 try: result[f"{surf_type}_L"] = gifti_to_trimesh(t1_gifti_paths[f"{surf_type}_L"])
                 except Exception as e: L.warning(f"Failed load T1 {t1_gifti_paths[f'{surf_type}_L']}: {e}")
                 try: result[f"{surf_type}_R"] = gifti_to_trimesh(t1_gifti_paths[f"{surf_type}_R"])
                 except Exception as e: L.warning(f"Failed load T1 {t1_gifti_paths[f'{surf_type}_R']}: {e}")

    except FileNotFoundError as e:
         L.error(f"Failed to find required cortical surface or transform file: {e}")
         # Fall through to structure extraction, result dict may be partially filled
    except Exception as e:
         L.error(f"Error processing cortical surfaces: {e}")
         # Fall through

    # --- Extract Additional Structures (CBM/BS/CC) ---
    for struct_name in extract_structures:
        label_ids = STRUCTURE_LABEL_MAP[struct_name]
        L.info(f"Extracting ASEG structure: {struct_name} in {space} space...")
        # Call helper function to extract the surface GII
        struct_gii_path = extract_structure_surface(
            subjects_dir=subjects_dir, subject_id=subject_id, # Pass original subject_id here
            label_ids=label_ids, output_tag=struct_name, space=space,
            tmp_dir=tmp_dir, verbose=verbose, session=session, run=run
        )
        # Load and process the GII file if created
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
            except Exception as e_mesh: L.warning(f"Failed load/process mesh {struct_name}: {e_mesh}"); result[struct_name] = None
        else: L.warning(f"Failed to generate surface file for {struct_name}"); result[struct_name] = None

    # --- Cleanup ---
    if local_tmp and temp_dir_obj:
        try: temp_dir_obj.cleanup(); L.info(f"Removed local temp dir => {tmp_dir}")
        except Exception as e_clean: L.warning(f"Failed remove temp dir {tmp_dir}: {e_clean}")

    return result


# --- 5ttgen Function (MODIFIED for work directory) ---

def run_5ttgen_hsvs_save_temp_bids(
    subject_id: str,
    fs_subject_dir: str, # Path to the input FreeSurfer subject directory.
    subject_work_dir: Union[str, Path], # MODIFIED: Path to the dedicated work dir for this subject/session
    # bids_root_dir: str, # REMOVED: No longer needed for BIDS derivative saving
    # pipeline_name: str = "brain_for_printing_5ttgen", # REMOVED: No longer managing BIDS structure here
    session_id: Optional[str] = None, # Kept for potential logging/internal use
    nocrop: bool = True,
    sgm_amyg_hipp: bool = True,
) -> Optional[str]:
    """
    Runs 5ttgen hsvs using a FreeSurfer directory as input, using a
    dedicated subject work directory for its scratch space, keeping the results.

    Args:
        subject_id (str): Subject identifier (e.g., "sub-01").
        fs_subject_dir (str): Path to the input FreeSurfer subject directory.
        subject_work_dir (Union[str, Path]): Path to the dedicated work directory
            for this subject/session (e.g., <work_dir>/brain_for_printing/sub-01/ses-A).
            This function assumes the base work directory structure already exists.
        session_id (Optional[str]): BIDS session ID (used primarily for logging now).
        nocrop (bool): Corresponds to the -nocrop option in 5ttgen.
        sgm_amyg_hipp (bool): Corresponds to the -sgm_amyg_hipp option in 5ttgen.

    Returns:
        Optional[str]: Path to the persistent 5ttgen working directory within the
                       subject_work_dir on success, or None on failure.
    """
    subject_work_dir = Path(subject_work_dir)
    subject_label_clean = subject_id.replace('sub-', '')
    sub_label = f"sub-{subject_label_clean}"
    ses_label = f"ses-{session_id}" if session_id else None

    # --- Define the persistent path within the subject's work directory ---
    persistent_5ttgen_path = subject_work_dir / "5ttgen_persistent_work"

    # --- Prepare and run 5ttgen command ---
    # Ensure the persistent directory exists before running 5ttgen
    try:
        persistent_5ttgen_path.mkdir(parents=True, exist_ok=True)
        L.info(f"Ensured persistent 5ttgen work directory exists: {persistent_5ttgen_path}")
    except Exception as e:
        L.error(f"Failed to create persistent 5ttgen work directory {persistent_5ttgen_path}: {e}")
        return None

    # Define the path for the intermediate 5TT output file (required by 5ttgen)
    # Store it inside the persistent path as well.
    temp_output_5tt = persistent_5ttgen_path / f"{sub_label}{'_'+ses_label if ses_label else ''}_5ttgen_output.nii.gz"

    # Build the 5ttgen command list
    cmd = ["5ttgen", "hsvs", fs_subject_dir, str(temp_output_5tt)] # Input is fs_subject_dir
    # Use the persistent path directly as the scratch directory
    cmd.extend(["-scratch", str(persistent_5ttgen_path)])
    if nocrop: cmd.append("-nocrop")
    if sgm_amyg_hipp: cmd.append("-sgm_amyg_hipp")
    cmd.append("-nocleanup") # Keep the contents of the scratch directory

    L.info(f"Running 5ttgen: {' '.join(cmd)}")
    try:
        # Execute the command
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        L.info("5ttgen hsvs completed successfully.")

        # --- Verify output ---
        # Check if the 5TT output file was created (basic check)
        if not temp_output_5tt.exists() or temp_output_5tt.stat().st_size == 0:
             L.warning(f"5ttgen main output file missing or empty: {temp_output_5tt}")
             # Decide if this is critical - might depend on whether the VTKs are needed
             # For now, we proceed assuming the VTK files might still exist

        # Check if the nested '5ttgen-tmp-*' directory was created inside persistent_5ttgen_path
        potential_nested_dirs = list(persistent_5ttgen_path.glob("5ttgen-tmp-*"))
        if not potential_nested_dirs:
            L.error(f"No '5ttgen-tmp-*' subdirectory found within {persistent_5ttgen_path} after 5ttgen run.")
            # Cleanup and return failure if the essential subdir is missing
            try:
                shutil.rmtree(persistent_5ttgen_path)
                L.info(f"Cleaned up persistent work directory on verification failure: {persistent_5ttgen_path}")
            except Exception as e_clean:
                L.error(f"Failed to clean up persistent work directory {persistent_5ttgen_path} after verification failure: {e_clean}")
            return None

        if len(potential_nested_dirs) > 1:
            L.warning(f"Multiple '5ttgen-tmp-*' subdirectories found in {persistent_5ttgen_path}. This is unexpected with -nocleanup.")
            # Proceeding with the first one found, but log a warning.

        L.info(f"5ttgen outputs retained in persistent work directory: {persistent_5ttgen_path}")
        return str(persistent_5ttgen_path) # Return the path on success

    except subprocess.CalledProcessError as e:
        L.error(f"5ttgen failed: {e.returncode}\nCMD: {' '.join(e.cmd)}\nSTDERR: {e.stderr}")
        # --- Post-5ttgen (Failure): Delete the entire persistent directory ---
        try:
            if persistent_5ttgen_path.exists():
                shutil.rmtree(persistent_5ttgen_path)
                L.info(f"Cleaned up persistent work directory on failure: {persistent_5ttgen_path}")
        except Exception as e_clean:
            L.error(f"Failed to clean up persistent work directory {persistent_5ttgen_path} after failure: {e_clean}")
        return None
    except Exception as e:
        L.error(f"Unexpected error during/after 5ttgen: {e}")
        # --- Post-5ttgen (Failure): Delete the entire persistent directory ---
        try:
            if persistent_5ttgen_path.exists():
                shutil.rmtree(persistent_5ttgen_path)
                L.info(f"Cleaned up persistent work directory on error: {persistent_5ttgen_path}")
        except Exception as e_clean:
            L.error(f"Failed to clean up persistent work directory {persistent_5ttgen_path} after error: {e_clean}")
        return None

    # Note: No BIDS dataset_description.json handling needed here anymore.

# Added import for numpy_support
from vtk.util import numpy_support
import vtk # Ensure vtk is imported

# --- Helper Function to read VTK using vtk library ---
def _read_vtk_polydata(path: str) -> Optional[vtk.vtkPolyData]:
    """Reads a VTK polydata file using vtkPolyDataReader."""
    try:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        if reader.GetErrorCode():
            L.error(f"VTK reader error code {reader.GetErrorCode()} for {path}")
            return None
        polydata = reader.GetOutput()
        if polydata and polydata.GetNumberOfPoints() > 0 and polydata.GetNumberOfCells() > 0:
            return polydata
        else:
            L.warning(f"VTK file seems empty or invalid: {path}")
            return None
    except Exception as e:
        L.error(f"Failed to read VTK file {path} using vtk library: {e}")
        return None

# --- Helper Function to convert vtkPolyData to Trimesh ---
def _vtk_polydata_to_trimesh(polydata: vtk.vtkPolyData) -> Optional[trimesh.Trimesh]:
    """Converts a vtk.vtkPolyData object to a trimesh.Trimesh object."""
    try:
        # Get vertices
        points = polydata.GetPoints()
        if not points: return None
        vertices = numpy_support.vtk_to_numpy(points.GetData())
        if vertices.shape[0] == 0: return None

        # Get faces (assuming triangles)
        polys = polydata.GetPolys()
        if not polys: return None
        faces_vtk = numpy_support.vtk_to_numpy(polys.GetData())

        # VTK faces array is like [n_verts1, v1_idx1, v1_idx2, ..., n_verts2, v2_idx1, ...]
        # We need to parse this assuming n_verts is always 3 for triangles
        if faces_vtk.shape[0] == 0: return None
        faces = []
        i = 0
        while i < len(faces_vtk):
            n_verts = faces_vtk[i]
            if n_verts == 3: # It's a triangle
                faces.append(faces_vtk[i+1 : i+1+n_verts])
            else:
                 L.warning(f"Non-triangular polygon encountered (num_vertices={n_verts}), skipping cell in VTK conversion.")
            i += (n_verts + 1) # Move to the next cell definition

        if not faces: return None
        # Ensure numpy is imported
        import numpy as np
        faces_np = np.array(faces, dtype=np.int64) # Use int64 for trimesh

        # Create Trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces_np)
        # Optional: Check and potentially repair before returning
        # mesh.process() # This can sometimes fail on complex meshes
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals() # Fix normals after potential repairs
        return mesh

    except Exception as e:
        L.error(f"Failed to convert vtkPolyData to Trimesh: {e}")
        return None


def load_subcortical_and_ventricle_meshes(five_ttgen_dir: Union[str, Path]) -> Dict[str, trimesh.Trimesh]:
    """
    Loads FSL FIRST (subcortical-*) and Ventricle/Vessel (ventricle-*, vessel-*)
    VTK files from a **persistent 5ttgen work directory**, searching within
    the nested '5ttgen-tmp-*' subdir. Converts to Trimesh objects.

    Args:
        five_ttgen_dir (Union[str, Path]): Path to the persistent 5ttgen work
                                          directory (e.g., .../5ttgen_persistent_work/).

    Returns:
        Dict[str, trimesh.Trimesh]: Dictionary mapping derived structure name
                                   (e.g., "subcortical-L_Thalamus", "ventricle-LatVent")
                                   to the loaded Trimesh object.
    """
    loaded_meshes: Dict[str, trimesh.Trimesh] = {}
    persistent_work_dir = Path(five_ttgen_dir) # Rename for clarity

    if not persistent_work_dir.is_dir():
        L.error(f"Provided persistent 5ttgen work directory not found: {persistent_work_dir}")
        return loaded_meshes

    # --- Find the actual data directory nested inside ---
    # 5ttgen with -nocleanup keeps files inside a '5ttgen-tmp-*' directory
    # within the specified scratch path.
    potential_nested_dirs = list(persistent_work_dir.glob("5ttgen-tmp-*"))
    if not potential_nested_dirs:
        L.error(f"Could not find the '5ttgen-tmp-*' subdirectory inside the persistent work dir: {persistent_work_dir}")
        return loaded_meshes

    if len(potential_nested_dirs) > 1:
        L.warning(f"Multiple '5ttgen-tmp-*' subdirs found in {persistent_work_dir}. Using the first: {potential_nested_dirs[0]}")

    search_dir = potential_nested_dirs[0] # This is where the VTK files actually are
    L.info(f"Searching for VTK files within: {search_dir}")

    skipped_count = 0

    # === Subcortical Structures (FSL FIRST output) ===
    # Updated glob pattern to search inside search_dir
    subcortical_pattern = str(search_dir / "first-*_transformed.vtk")
    raw_subcortical_glob = glob.glob(subcortical_pattern)
    # Filtering logic remains the same
    subcortical_candidates = [
        p for p in raw_subcortical_glob
        if not Path(p).name.endswith("_first_transformed.vtk")
    ]
    L.info(f"Found {len(subcortical_candidates)} potential FIRST subcortical VTK files (after filtering).")

    for vtk_path_str in subcortical_candidates:
        vtk_path = Path(vtk_path_str); struct_name = None
        # Naming logic remains the same
        filename = vtk_path.name
        if filename.startswith("first-") and filename.endswith("_transformed.vtk"):
             name_part = filename[len("first-"):-len("_transformed.vtk")]
             struct_name = f"subcortical-{name_part}"
        else:
             L.warning(f"Unexpected subcortical filename format: {filename}. Skipping naming.")
             struct_name = f"subcortical-unknown-{vtk_path.stem}"

        L.debug(f"Attempting to load subcortical VTK: {vtk_path.name} as {struct_name} using vtk library")

        polydata = _read_vtk_polydata(vtk_path_str)
        if polydata:
            mesh = _vtk_polydata_to_trimesh(polydata)
            if mesh and not mesh.is_empty:
                loaded_meshes[struct_name] = mesh
                L.debug(f"Successfully loaded and converted {struct_name}")
            else:
                L.warning(f"Failed to convert {struct_name} from vtkPolyData to Trimesh or result was empty.")
                skipped_count += 1
        else:
            skipped_count += 1

    # === Ventricles, Choroid Plexus, Vessels ===
    ventricle_tags = ["Ventricle", "LatVent", "ChorPlex", "Inf-Lat-Vent", "vessel"]
    # Updated glob pattern to search inside search_dir
    all_vtk_pattern = str(search_dir / "*.vtk")
    raw_ventricle_glob = glob.glob(all_vtk_pattern)
    # Filtering logic remains the same
    ventricle_candidates = [
        f for f in raw_ventricle_glob
        if any(tag in Path(f).name for tag in ventricle_tags)
        and not Path(f).name.endswith("_init.vtk")
        and "_first" not in Path(f).name
    ]
    L.info(f"Found {len(ventricle_candidates)} potential Ventricle/Vessel VTK files (after filtering).")

    for vtk_path_str in ventricle_candidates:
        vtk_path = Path(vtk_path_str); struct_name = None
        # Naming logic remains the same
        base_name = vtk_path.stem
        struct_prefix = "vessel" if "vessel" in base_name.lower() else "ventricle"
        struct_name = f"{struct_prefix}-{base_name}"

        L.debug(f"Attempting to load ventricle/vessel VTK: {vtk_path.name} as {struct_name} using vtk library")
        polydata = _read_vtk_polydata(vtk_path_str)
        if polydata:
            mesh = _vtk_polydata_to_trimesh(polydata)
            if mesh and not mesh.is_empty:
                 # Optional: Invert normals if needed for these structures
                 # L.debug(f"Inverting normals for {struct_name}"); mesh.invert()
                 loaded_meshes[struct_name] = mesh
                 L.debug(f"Successfully loaded and converted {struct_name}")
            else:
                 L.warning(f"Failed to convert {struct_name} from vtkPolyData to Trimesh or result was empty.")
                 skipped_count += 1
        else:
            skipped_count += 1

    if skipped_count > 0: L.warning(f"Skipped loading/converting {skipped_count} VTK files.")
    L.info(f"Loaded {len(loaded_meshes)} VTK-derived meshes in total.")
    return loaded_meshes

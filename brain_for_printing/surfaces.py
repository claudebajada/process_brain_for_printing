"""
surfaces.py
-----------
Generate multiple brain surfaces (cortical + optional brainstem) in T1 or MNI space,
including pial, white (smoothwm), or mid (midthickness) surfaces for each hemisphere.
Also extract an optional brainstem surface.

Usage:
  from .surfaces import generate_brain_surfaces
  meshes_dict = generate_brain_surfaces(
      subjects_dir="/path/to/derivatives",
      subject_id="sub-01",
      space="T1",
      surfaces=("pial", "mid", "white"),
      no_brainstem=False
  )
"""
import os
import uuid
import shutil
import trimesh

from .io_utils import run_cmd, flexible_match, first_match
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .warp_utils import generate_mrtrix_style_warp, warp_gifti_vertices
from .constants import BRAINSTEM_LABELS

def extract_brainstem_in_t1(subjects_dir, subject_id,
                            tmp_dir=".", verbose=False,
                            session=None, run=None):
    """
    Locate and binarize Freesurfer's aseg in T1 space, isolating the 'brainstem' region,
    then convert to GIFTI with marching_cubes. Returns a .surf.gii file path.
    """
    anat_dir = os.path.join(subjects_dir, subject_id, "anat")

    aseg_nii = flexible_match(
        base_dir=anat_dir,
        subject_id=subject_id,
        descriptor="desc-aseg",
        suffix="dseg",
        ext=".nii.gz"
    )

    t1_brainstem_tmp  = os.path.join(tmp_dir, "brainstem_tmp_T1.nii.gz")
    run_cmd(["mri_convert", aseg_nii, t1_brainstem_tmp], verbose=verbose)

    final_mask_nii = os.path.join(tmp_dir, "brainstem_bin_T1.nii.gz")
    match_str = [str(lbl) for lbl in BRAINSTEM_LABELS]

    run_cmd([
        "mri_binarize",
        "--i", t1_brainstem_tmp,
        "--match"
    ] + match_str + [
        "--inv",
        "--o", os.path.join(tmp_dir, "bin_tmp_T1.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", t1_brainstem_tmp, "-mul",
        os.path.join(tmp_dir, "bin_tmp_T1.nii.gz"),
        os.path.join(tmp_dir, "brainstem_mask_T1.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", os.path.join(tmp_dir, "brainstem_mask_T1.nii.gz"),
        "-bin", final_mask_nii
    ], verbose=verbose)

    out_gii = os.path.join(tmp_dir, "brainstem_in_t1.surf.gii")
    volume_to_gifti(final_mask_nii, out_gii, level=0.5)
    return out_gii


def extract_brainstem_in_mni(subjects_dir, subject_id,
                             out_aseg_in_mni, tmp_dir=".", verbose=False,
                             session=None, run=None):
    """
    Warp the Freesurfer aseg from T1 to MNI, binarize, and convert to GIFTI.
    Returns a .surf.gii file path.
    """
    anat_dir = os.path.join(subjects_dir, subject_id, "anat")

    def build_pattern(descriptor=None, suffix=None):
        pattern = f"{anat_dir}/{subject_id}"
        if session:
            pattern += f"_{session}"
        if run:
            pattern += f"_{run}"
        if descriptor:
            pattern += f"*{descriptor}*"
        if suffix:
            pattern += f"*{suffix}"
        pattern += ".nii.gz"
        return pattern

    aseg_nii = first_match(build_pattern("desc-aseg", "dseg"))
    xfm_file = first_match(build_pattern("from-T1w_to-MNI152NLin2009cAsym_mode-image", "xfm").replace(".nii.gz", ".h5"))
    mni_template = first_match(build_pattern("space-MNI152NLin2009cAsym", "T1w"))

    if verbose:
        print(f"[INFO] aseg matched => {aseg_nii}")
        print(f"[INFO] xfm matched  => {xfm_file}")
        print(f"[INFO] mni_template => {mni_template}")

    run_cmd(["antsApplyTransforms", "-d", "3", "-i", aseg_nii, "-o", out_aseg_in_mni, "-r", mni_template, "-t", xfm_file, "-n", "NearestNeighbor"], verbose=verbose)

    final_mask_nii = os.path.join(tmp_dir, "brainstem_bin_mni.nii.gz")
    tmp_nii = os.path.join(tmp_dir, "brainstem_tmp_mni.nii.gz")
    run_cmd(["mri_convert", out_aseg_in_mni, tmp_nii], verbose=verbose)

    match_str = [str(lbl) for lbl in BRAINSTEM_LABELS]
    run_cmd(["mri_binarize", "--i", tmp_nii, "--match"] + match_str + ["--inv", "--o", os.path.join(tmp_dir, "bin_tmp_mni.nii.gz")], verbose=verbose)
    run_cmd(["fslmaths", tmp_nii, "-mul", os.path.join(tmp_dir, "bin_tmp_mni.nii.gz"), os.path.join(tmp_dir, "brainstem_mask_mni.nii.gz")], verbose=verbose)
    run_cmd(["fslmaths", os.path.join(tmp_dir, "brainstem_mask_mni.nii.gz"), "-bin", final_mask_nii], verbose=verbose)

    out_gii = os.path.join(tmp_dir, "brainstem_in_mni.surf.gii")
    volume_to_gifti(final_mask_nii, out_gii, level=0.5)
    return out_gii


# ---------------------------------------
# GENERATE MULTIPLE BRAIN SURFACES
# ---------------------------------------
def generate_brain_surfaces(
    subjects_dir,
    subject_id,
    space="T1",
    surfaces=("pial",),
    no_brainstem=False,
    no_fill=False,
    no_smooth=False,
    out_warp="warp.nii",
    run=None,
    session=None,
    verbose=False,
    tmp_dir=None
) -> dict:
    """
    Generate multiple brain surfaces (cortical + optional brainstem) in T1 or MNI space.
    
    This can include pial, mid-thickness, and/or white matter surfaces for each hemisphere,
    as well as an optional brainstem. All surfaces are returned as Trimesh objects
    in a single dictionary. If 'space' is MNI, surfaces are automatically warped via
    ANTs+MRtrix methods.

    Args:
        subjects_dir (str): Path to derivatives or subject data (with /anat subfolder).
        subject_id (str): Subject identifier.
        space (str): "T1" (native) or "MNI" (warped).
        surfaces (tuple of str): Which surfaces to load (e.g. ("pial","mid","white")).
                                 We map "pial" => "_pial.surf.gii",
                                              "mid"  => "_midthickness.surf.gii",
                                              "white"=> "_smoothwm.surf.gii"
        no_brainstem (bool): If True, skip extracting the brainstem.
        no_fill (bool): Skip hole-filling in extracted brainstem.
        no_smooth (bool): Skip smoothing in extracted brainstem.
        out_warp (str): 4D warp field filename if warping to MNI.
        run (str): BIDS run ID, e.g., "run-01" (optional).
        session (str): BIDS session ID, e.g., "ses-01" (optional).
        verbose (bool): If True, print extra info.
        tmp_dir (str): If provided, use this folder for intermediate files;
                       otherwise a new one is created and removed.

    Returns:
        dict: Trimesh objects keyed by surface type:
            {
                "pial_L":  <Trimesh or None>,
                "pial_R":  <Trimesh or None>,
                "mid_L":   <Trimesh or None>,
                "mid_R":   <Trimesh or None>,
                "white_L": <Trimesh or None>,
                "white_R": <Trimesh or None>,
                "brainstem": <Trimesh or None>
            }
        Surfaces not requested in 'surfaces' will be None.
        If no_brainstem=True, "brainstem" will be None.
    """

    # Map user-requested types to actual BIDS suffixes in your directory
    SURF_NAME_MAP = {
        "pial": "pial",
        "mid": "midthickness",
        "white": "smoothwm"
    }

    # If user gave something unexpected, just fall back to exactly that string
    # but usually pial -> pial.surf, mid -> midthickness.surf, white -> smoothwm.surf
    # is correct for your data set
    def resolve_suffix(surf_type):
        return SURF_NAME_MAP.get(surf_type, surf_type)

    local_tmp = False
    if not tmp_dir:
        tmp_dir = f"_tmp_surf_{uuid.uuid4().hex[:6]}"
        os.makedirs(tmp_dir, exist_ok=True)
        local_tmp = True
        if verbose:
            print(f"[INFO] Created local temp dir => {tmp_dir}")

    anat_dir = os.path.join(subjects_dir, subject_id, "anat")

    # Prepare our output dict with placeholders
    result = {
        "pial_L": None,
        "pial_R": None,
        "mid_L": None,
        "mid_R": None,
        "white_L": None,
        "white_R": None,
        "brainstem": None
    }

    # ----------------------------------------------------------------
    # Identify T1 GIFTIs for each requested surface type
    # ----------------------------------------------------------------
    t1_gifti_paths = {}
    for surf_type in surfaces:  # e.g. "pial", "mid", or "white"
        # Map to your actual file naming:
        actual_name = resolve_suffix(surf_type)  # e.g. "midthickness"
        lh_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor=None,
            suffix=f"{actual_name}.surf",
            hemi="hemi-L",
            ext=".gii",
            run=run,
            session=session
        )
        rh_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor=None,
            suffix=f"{actual_name}.surf",
            hemi="hemi-R",
            ext=".gii",
            run=run,
            session=session
        )
        t1_gifti_paths[f"{surf_type}_L"] = lh_file
        t1_gifti_paths[f"{surf_type}_R"] = rh_file

    # ----------------------------------------------------------------
    # If space=MNI, warp each GIFTI to MNI
    # ----------------------------------------------------------------
    if space.upper() == "MNI":
        def build_pattern(descriptor=None, suffix=None):
            pattern = f"{anat_dir}/{subject_id}"
            if session:
                pattern += f"_{session}"
            if run:
                pattern += f"_{run}"
            if descriptor:
                pattern += f"*{descriptor}*"
            if suffix:
                pattern += f"*{suffix}"
            pattern += ".nii.gz"
            return pattern

        mni_file = first_match(build_pattern("space-MNI152NLin2009cAsym", "T1w"))
        t1_file = first_match(build_pattern("desc-preproc", "T1w"))
        xfm_file = first_match(build_pattern("from-MNI152NLin2009cAsym_to-T1w_mode-image", "xfm").replace(".nii.gz", ".h5"))
        
        if verbose:
            print(f"[INFO] MNI file matched     => {mni_file}")
            print(f"[INFO] T1 preproc matched   => {t1_file}")
            print(f"[INFO] Transform (xfm) file => {xfm_file}")


        warp_field = os.path.join(tmp_dir, out_warp)
        generate_mrtrix_style_warp(mni_file, t1_file, xfm_file, out_warp, tmp_dir, verbose)

        # For each requested surface, warp LH and RH
        for surf_type in surfaces:
            lh_out = os.path.join(tmp_dir, f"L_{surf_type}_mni.gii")
            rh_out = os.path.join(tmp_dir, f"R_{surf_type}_mni.gii")
            warp_gifti_vertices(t1_gifti_paths[f"{surf_type}_L"], warp_field, lh_out, verbose=verbose)
            warp_gifti_vertices(t1_gifti_paths[f"{surf_type}_R"], warp_field, rh_out, verbose=verbose)

            result[f"{surf_type}_L"] = gifti_to_trimesh(lh_out)
            result[f"{surf_type}_R"] = gifti_to_trimesh(rh_out)

    else:
        # T1: directly load GIFTIs
        for surf_type in surfaces:
            lh_path = t1_gifti_paths[f"{surf_type}_L"]
            rh_path = t1_gifti_paths[f"{surf_type}_R"]
            result[f"{surf_type}_L"] = gifti_to_trimesh(lh_path)
            result[f"{surf_type}_R"] = gifti_to_trimesh(rh_path)

    # ----------------------------------------------------------------
    # Extract brainstem if requested
    # ----------------------------------------------------------------
    if not no_brainstem:
        if space.upper() == "T1":
            bs_gii = extract_brainstem_in_t1(
                subjects_dir=subjects_dir,
                subject_id=subject_id,
                tmp_dir=tmp_dir,
                verbose=verbose
            )
        else:
            out_aseg_in_mni = os.path.join(tmp_dir, "aseg_in_mni.nii.gz")
            bs_gii = extract_brainstem_in_mni(
                subjects_dir=subjects_dir,
                subject_id=subject_id,
                out_aseg_in_mni=out_aseg_in_mni,
                tmp_dir=tmp_dir,
                verbose=verbose,
                run=run,
                session=session
            )

        bs_mesh = gifti_to_trimesh(bs_gii)
        if not no_fill:
            trimesh.repair.fill_holes(bs_mesh)
        if not no_smooth:
            trimesh.smoothing.filter_taubin(bs_mesh, lamb=0.5, nu=-0.53, iterations=10)
        bs_mesh.invert()

        result["brainstem"] = bs_mesh

    # ----------------------------------------------------------------
    # Cleanup local tmp if we created it here
    # ----------------------------------------------------------------
    if local_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if verbose:
            print(f"[INFO] Removed local temp dir => {tmp_dir}")

    return result


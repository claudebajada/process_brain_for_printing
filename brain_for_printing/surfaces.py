# brain_for_printing/surfaces.py

import os
import uuid
import shutil
import trimesh

from .io_utils import run_cmd, flexible_match
from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .warp_utils import generate_mrtrix_style_warp, warp_gifti_vertices

# ---------------------------------------
# BRAINSTEM LABELS & EXTRACTION FUNCTIONS
# ---------------------------------------
BRAINSTEM_LABELS = [
    2,3,24,31,41,42,63,72,77,51,52,13,12,
    43,50,4,11,26,58,49,10,17,18,53,54,
    44,5,80,14,15,30,62
]

def extract_brainstem_in_t1(subjects_dir, subject_id,
                            tmp_dir=".", verbose=False):
    """
    Locate and binarize Freesurfer's aseg in T1 space, isolating 'brainstem' region,
    then convert to GIFTI with marching_cubes. Surfaces are stored temporarily and
    returned as a .surf.gii file path.
    """
    # This import or function depends on your pipeline naming conventions
    # The user might have fMRIPrep aseg: sub-XX_desc-aseg_dseg.nii.gz etc.
    # Adjust if needed.
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

    # Binarize labels -> invert
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

    aseg_nii = flexible_match(
        base_dir=anat_dir,
        subject_id=subject_id,
        descriptor="desc-aseg",
        suffix="dseg",
        session=session,
        run=run,
        ext=".nii.gz"
    )

    xfm_file = flexible_match(
        base_dir=anat_dir,
        subject_id=subject_id,
        descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image",
        suffix="xfm",
        session=session,
        run=run,
        ext=".h5"
    )

    mni_template = flexible_match(
        base_dir=anat_dir,
        subject_id=subject_id,
        descriptor="space-MNI152NLin2009cAsym",
        suffix="T1w",
        session=session,
        run=run,
        ext=".nii.gz"
    )

    # Warp aseg -> MNI
    run_cmd([
        "antsApplyTransforms", "-d", "3",
        "-i", aseg_nii,
        "-o", out_aseg_in_mni,
        "-r", mni_template,
        "-t", xfm_file,
        "-n", "NearestNeighbor"
    ], verbose=verbose)

    final_mask_nii = os.path.join(tmp_dir, "brainstem_bin_mni.nii.gz")
    tmp_nii        = os.path.join(tmp_dir, "brainstem_tmp_mni.nii.gz")
    run_cmd(["mri_convert", out_aseg_in_mni, tmp_nii], verbose=verbose)

    match_str = [str(lbl) for lbl in BRAINSTEM_LABELS]
    run_cmd([
        "mri_binarize",
        "--i", tmp_nii,
        "--match"
    ] + match_str + [
        "--inv",
        "--o", os.path.join(tmp_dir, "bin_tmp_mni.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", tmp_nii, "-mul",
        os.path.join(tmp_dir, "bin_tmp_mni.nii.gz"),
        os.path.join(tmp_dir, "brainstem_mask_mni.nii.gz")
    ], verbose=verbose)

    run_cmd([
        "fslmaths", os.path.join(tmp_dir, "brainstem_mask_mni.nii.gz"),
        "-bin", final_mask_nii
    ], verbose=verbose)

    out_gii = os.path.join(tmp_dir, "brainstem_in_mni.surf.gii")
    volume_to_gifti(final_mask_nii, out_gii, level=0.5)
    return out_gii


# ---------------------------------------
# GENERATE MULTIPLE BRAIN SURFACES (Pial, Mid, White) + Brainstem
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
            Each must match a GIFTI suffix like "*_hemi-L_pial.surf.gii".
        no_brainstem (bool): If True, skip extracting the brainstem.
        no_fill (bool): Skip hole-filling in extracted brainstem.
        no_smooth (bool): Skip smoothing in extracted brainstem.
        out_warp (str): 4D warp field filename if warping to MNI.
        run (str): BIDS run ID, e.g., "run-01" (optional).
        session (str): BIDS session ID, e.g., "ses-01" (optional).
        verbose (bool): If True, print extra info.
        tmp_dir (str): If provided, use this existing folder for intermediate files;
                       otherwise a new one is created and cleaned up internally.

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

    # Decide whether we create a temporary folder ourselves
    local_tmp = False
    if not tmp_dir:
        tmp_dir = f"_tmp_surf_{uuid.uuid4().hex[:6]}"
        os.makedirs(tmp_dir, exist_ok=True)
        local_tmp = True
        if verbose:
            print(f"[INFO] Created local temp dir => {tmp_dir}")

    anat_dir = os.path.join(subjects_dir, subject_id, "anat")

    # We'll store final meshes in a dictionary
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
    # 1) Identify T1 GIFTIs for each requested surface type
    # ----------------------------------------------------------------
    t1_gifti_paths = {}
    for surf_type in surfaces:  # e.g. "pial", "mid", "white"
        lh_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor=None,
            suffix=f"{surf_type}.surf",
            hemi="hemi-L",
            ext=".gii",
            run=run,
            session=session
        )
        rh_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor=None,
            suffix=f"{surf_type}.surf",
            hemi="hemi-R",
            ext=".gii",
            run=run,
            session=session
        )
        t1_gifti_paths[f"{surf_type}_L"] = lh_file
        t1_gifti_paths[f"{surf_type}_R"] = rh_file

    # ----------------------------------------------------------------
    # 2) If space=MNI, warp each GIFTI
    # ----------------------------------------------------------------
    if space.upper() == "MNI":
        # Generate or locate warp field
        mni_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor="space-MNI152NLin2009cAsym",
            suffix="T1w",
            run=run,
            session=session,
            ext=".nii.gz"
        )
        t1_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor="desc-preproc",
            suffix="T1w",
            run=run,
            session=session,
            ext=".nii.gz"
        )
        xfm_file = flexible_match(
            base_dir=anat_dir,
            subject_id=subject_id,
            descriptor="from-MNI152NLin2009cAsym_to-T1w_mode-image",
            suffix="xfm",
            run=run,
            session=session,
            ext=".h5"
        )

        warp_field = os.path.join(tmp_dir, out_warp)
        generate_mrtrix_style_warp(
            mni_file=mni_file,
            t1_file=t1_file,
            xfm_file=xfm_file,
            out_warp=out_warp,
            tmp_dir=tmp_dir,
            verbose=verbose
        )

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
    # 3) Extract brainstem if requested
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
    # 4) Cleanup local tmp if needed
    # ----------------------------------------------------------------
    if local_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if verbose:
            print(f"[INFO] Removed local temp dir => {tmp_dir}")

    return result

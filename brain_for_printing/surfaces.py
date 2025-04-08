# brain_for_printing/surfaces.py

import os
from .io_utils import run_cmd, first_match
from .mesh_utils import volume_to_gifti, gifti_to_trimesh


# Check whether these BRAINSTEM_LABELS are the ones to keep, remove '--inv' 
# in mri_binarize. If they're the ones to exclude, keep '--inv'.
# A note that fmriprep must have run the MNI152NLin2009cAsym space
    
BRAINSTEM_LABELS = [2,3,24,31,41,42,63,72,77,51,52,13,12,
                    43,50,4,11,26,58,49,10,17,18,53,54,
                    44,5,80,14,15,30,62]

def extract_brainstem_in_t1(subjects_dir, subject_id,
                            tmp_dir=".", verbose=False):
    """
    Locate freesurfer aseg in T1 space, isolate 'brainstem' region,
    convert to GIFTI with marching_cubes.

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
                             out_aseg_in_mni, tmp_dir=".", verbose=False, session=None, run=None):
    """
    Warp the aseg from T1 to MNI, binarize, and convert to GIFTI.

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


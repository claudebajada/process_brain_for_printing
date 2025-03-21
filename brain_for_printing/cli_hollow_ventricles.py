# brain_for_printing/cli_hollow_ventricles.py

import os
import argparse
import uuid
import shutil
import trimesh

from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .io_utils import first_match, run_cmd

def extract_ventricles_mask(subjects_dir, subject_id, space, tmp_dir, verbose=False):
    """
    Extract ventricular structures from fMRIPrep aseg.
    """
    anat_dir = os.path.join(subjects_dir, subject_id, "anat")
    if space == "T1":
        aseg_pattern = f"{anat_dir}/*_run-01_desc-aseg_dseg.nii.gz"
    else:
        aseg_pattern = f"{anat_dir}/*_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz"

    aseg_file = first_match(aseg_pattern)
    aseg_tmp = os.path.join(tmp_dir, f"aseg_{space}.nii.gz")
    run_cmd(["mri_convert", aseg_file, aseg_tmp], verbose=verbose)

    # Labels for ventricles
    vent_labels = [4, 5, 14, 15, 43, 44, 72]  # Lat, inf lat, 3rd, 4th, CSF

    vent_mask_tmp = os.path.join(tmp_dir, f"vent_mask_{space}.nii.gz")
    run_cmd(["mri_binarize", "--i", aseg_tmp, "--match"] + [str(l) for l in vent_labels] + ["--o", vent_mask_tmp], verbose=verbose)

    return vent_mask_tmp

def main():
    parser = argparse.ArgumentParser(
        description="Hollow out the ventricles from a brain STL using fMRIPrep aseg outputs."
    )
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--in_mesh", required=True, help="Input brain mesh STL file.")
    parser.add_argument("--space", choices=["T1", "MNI"], default="T1")
    parser.add_argument("--output", default="brain_hollowed.stl")
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tmp_dir = os.path.join(os.path.dirname(args.output), f"_tmp_hollow_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)

    # Step 1: Load brain mesh
    print(f"[INFO] Loading brain mesh => {args.in_mesh}")
    brain_mesh = trimesh.load(args.in_mesh)

    # Step 2: Extract ventricles mask and convert to GIFTI
    vent_mask = extract_ventricles_mask(args.subjects_dir, args.subject_id, args.space, tmp_dir, verbose=args.verbose)
    vent_gii = os.path.join(tmp_dir, "ventricles.surf.gii")
    volume_to_gifti(vent_mask, vent_gii, level=0.5)

    # Step 3: Convert ventricles mesh and subtract
    vent_mesh = gifti_to_trimesh(vent_gii)
    vent_mesh = vent_mesh.convex_hull  # ensure watertight

    print("[INFO] Performing boolean subtraction (brain - ventricles)...")
    hollowed = trimesh.boolean.difference([brain_mesh], [vent_mesh])

    if not hollowed or hollowed.is_empty:
        raise RuntimeError("Boolean subtraction failed or produced empty mesh.")

    # Step 4: Export result
    hollowed.export(args.output)
    print(f"[INFO] Exported => {args.output}")

    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")

if __name__ == "__main__":
    main()

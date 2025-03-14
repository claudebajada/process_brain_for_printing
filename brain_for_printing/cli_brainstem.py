# brain_for_printing/cli_brainstem.py

import os
import argparse
import uuid
import shutil
from .surfaces import extract_brainstem_in_t1, extract_brainstem_in_mni
from .mesh_utils import gifti_to_trimesh
import trimesh

def main():
    parser = argparse.ArgumentParser(
        description="Extract brainstem-only surfaces (T1 or MNI)."
    )
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--space", choices=["T1","MNI"], default="T1")
    parser.add_argument("--no_fill", action="store_true")
    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--no_clean", action="store_true")
    args = parser.parse_args()

    do_fill = not args.no_fill
    do_smooth = not args.no_smooth
    do_clean = not args.no_clean

    tmp_dir = os.path.join(args.output_dir,
                           f"_tmp_brainstem_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)

    if args.space.upper() == "T1":
        bs_gii = extract_brainstem_in_t1(args.subjects_dir, args.subject_id,
                                         tmp_dir=tmp_dir)
    else:
        out_aseg_in_mni = os.path.join(tmp_dir, "aseg_in_mni.nii.gz")
        bs_gii = extract_brainstem_in_mni(
            args.subjects_dir, args.subject_id,
            out_aseg_in_mni=out_aseg_in_mni,
            tmp_dir=tmp_dir
        )

    # Post-process
    bs_mesh = gifti_to_trimesh(bs_gii)
    if do_fill:
        trimesh.repair.fill_holes(bs_mesh)
    if do_smooth:
        trimesh.smoothing.filter_taubin(bs_mesh, lamb=0.5, nu=-0.53, iterations=10)
    bs_mesh.invert()

    out_stl = os.path.join(args.output_dir,
                           f"{args.subject_id}_brainstem_{args.space}.stl")
    bs_mesh.export(out_stl, file_type="stl")

    if do_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)


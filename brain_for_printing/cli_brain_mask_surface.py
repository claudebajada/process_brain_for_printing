#!/usr/bin/env python
# brain_for_printing/cli_brain_mask_surface.py
#
# Generate a surface from a T1‑brain mask (T1 or MNI), optionally inflate and
# smooth it, then export as STL.  Uses the new logging & temp‑folder helpers.

import argparse
import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, generate_binary_structure

from .io_utils import (flexible_match, run_cmd, temp_dir, require_cmd)
from .log_utils import write_log, get_logger
from .mesh_utils import volume_to_gifti, gifti_to_trimesh

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Create a smooth (optionally inflated) surface from a T1‑brain mask "
            "and export it as STL.  If --space MNI is chosen, the mask is first "
            "warped using ANTs."
        )
    )
    p.add_argument("--subjects_dir", required=True)
    p.add_argument("--subject_id",  required=True)
    p.add_argument("--output_dir",  default=".")
    p.add_argument("--space",      choices=["T1", "MNI"], default="T1")
    p.add_argument("--inflate_mm", type=float, default=0.0,
                   help="Dilate the binary mask by N mm before meshing.")
    p.add_argument("--no_smooth", action="store_true")
    p.add_argument("--no_clean",  action="store_true")
    p.add_argument("--run",     default=None)
    p.add_argument("--session", default=None)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print INFO‑level log messages.")

    args = p.parse_args()

    # ------------------------  real‑time logger  --------------------------- #
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)
    L.info("Starting brain_mask_surface…")

    # -----------------------------  checks  ------------------------------- #
    if args.space == "MNI":
        require_cmd("antsApplyTransforms", "https://github.com/ANTsX/ANTs")  # step 3 helper

    # ---------------------------  run‑log dict  --------------------------- #
    runlog = {
        "tool":        "brain_for_printing_brain_mask_surface",
        "subject_id":  args.subject_id,
        "space":       args.space,
        "inflate_mm":  args.inflate_mm,
        "steps":       [],
        "warnings":    [],
        "output_dir":  os.path.abspath(args.output_dir),
        "output_files": []
    }

    anat_dir = Path(args.subjects_dir) / args.subject_id / "anat"

    # Locate T1 brain‑mask
    mask_file = flexible_match(
        base_dir  = anat_dir,
        subject_id= args.subject_id,
        descriptor= "desc-brain",
        suffix   = "mask",
        session  = args.session,
        run      = args.run,
        ext      = ".nii.gz"
    )
    runlog["mask_file"] = mask_file
    runlog["steps"].append("Located T1w brain mask")

    # -----------------------  work in a temp folder  ---------------------- #
    with temp_dir("masksurf") as tmp_dir:
        L.info(f"Working directory: {tmp_dir}")

        # Warp to MNI if requested
        final_mask = mask_file
        if args.space == "MNI":
            xfm = flexible_match(anat_dir, args.subject_id,
                                 descriptor="from-T1w_to-MNI152NLin2009cAsym_mode-image",
                                 suffix="xfm",
                                 session=args.session, run=args.run, ext=".h5")
            template = flexible_match(anat_dir, args.subject_id,
                                      descriptor="space-MNI152NLin2009cAsym",
                                      suffix="T1w",
                                      session=args.session, run=args.run)

            final_mask = tmp_dir / "brain_mask_mni.nii.gz"
            run_cmd([
                "antsApplyTransforms", "-d", "3",
                "-i", mask_file,
                "-o", str(final_mask),
                "-r", template,
                "-t", xfm,
                "-n", "NearestNeighbor"
            ], verbose=args.verbose)
            runlog["steps"].append("Warped brain mask to MNI")

        # Optional inflation
        if args.inflate_mm > 0:
            nii   = nib.load(final_mask)
            data  = nii.get_fdata() > 0
            aff   = nii.affine
            vox   = np.sqrt((aff[:3, :3] ** 2).sum(axis=0)).mean()
            steps = int(np.ceil(args.inflate_mm / vox))
            L.info(f"Inflating by {args.inflate_mm} mm ≈ {steps} vox")

            struct  = generate_binary_structure(3, 1)
            dilated = data.copy()
            for _ in range(steps):
                dilated = binary_dilation(dilated, structure=struct)

            inflated = tmp_dir / "inflated_mask.nii.gz"
            nib.save(nib.Nifti1Image(dilated.astype(np.uint8), aff), inflated)
            runlog["steps"].append(f"Inflated mask by {args.inflate_mm} mm")
            final_mask = inflated

        # -----------------------  mask → mesh ----------------------------- #
        gii_path = tmp_dir / "mask_surface.gii"
        volume_to_gifti(str(final_mask), str(gii_path), level=0.5)
        runlog["steps"].append("Converted mask to GIFTI")

        mesh = gifti_to_trimesh(str(gii_path))
        if not args.no_smooth:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
            runlog["steps"].append("Applied Taubin smoothing")

        mesh.invert()                     # normals outward
        runlog["steps"].append("Inverted normals")

        out_stl = Path(args.output_dir) / f"{args.subject_id}_{args.space}_mask_surface.stl"
        mesh.export(out_stl, file_type="stl")
        runlog["output_files"].append(str(out_stl))
        L.info(f"Exported STL → {out_stl}")

        # Optionally keep the temp dir
        if args.no_clean:
            L.warning(f"Temporary folder retained at {tmp_dir}")
            runlog["warnings"].append("Temp folder retained by --no_clean")
        # ─────────────────────────────────────────────────────────────────── #

    # -----------------------  write JSON log  ----------------------------- #
    write_log(runlog, args.output_dir, base_name="mask_surface_log")
    L.info("Done.")


if __name__ == "__main__":
    main()

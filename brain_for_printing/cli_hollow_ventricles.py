#!/usr/bin/env python
# brain_for_printing/cli_hollow_ventricles.py
#
# Subtract the ventricular system from a brain mesh using fMRIPrep’s aseg
# segmentation.  Supports optional mask dilation and works in either T1 or MNI
# space (T1 implemented here).  Implements structured logging, graceful temp‑
# dir handling, and external‑tool checks.

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from .io_utils import (
    flexible_match,
    first_match,
    run_cmd,
    require_cmd,
    temp_dir,
)
from .mesh_utils import (
    volume_to_gifti,
    gifti_to_trimesh,
    voxel_remesh_and_repair,
)
from .log_utils import get_logger, write_log


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    pa = argparse.ArgumentParser(
        description="Subtract ventricles from a brain mesh using fMRIPrep aseg."
    )
    pa.add_argument("--subjects_dir", required=True, help="BIDS derivatives root.")
    pa.add_argument("--subject_id", required=True, help="e.g. sub‑01")
    pa.add_argument("--in_mesh", required=True, help="Path to brain STL/OBJ/GIFTI.")
    pa.add_argument("--space", choices=["T1", "MNI"], default="T1")
    pa.add_argument("--output", required=True, help="Output STL path.")
    pa.add_argument(
        "--engine",
        choices=["scad", "blender", "auto"],
        default="auto",
        help="Boolean backend for trimesh.",
    )
    pa.add_argument(
        "--dilate_mask",
        action="store_true",
        help="Dilate + erode ventricle mask before meshing (helps integrity).",
    )
    pa.add_argument("--run", default=None, help="BIDS run‑label if present.")
    pa.add_argument("--session", default=None, help="BIDS session‑label if present.")
    pa.add_argument("--no_clean", action="store_true", help="Keep temp folder.")
    pa.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print INFO‑level messages and external‑command output.",
    )
    return pa


def main() -> None:
    args = _build_parser().parse_args()

    # ---------- logger ----------
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)

    # ---------- external tools ----------
    require_cmd(
        "mri_binarize",
        "https://surfer.nmr.mgh.harvard.edu/",
        logger=L,
    )
    if args.engine == "blender":
        require_cmd("blender", "https://www.blender.org/", logger=L)
    elif args.engine == "scad":
        require_cmd("openscad", "https://openscad.org/", logger=L)

    # ---------- structured JSON log ----------
    runlog = {
        "tool": "brain_for_printing_hollow_ventricles",
        "subject_id": args.subject_id,
        "space": args.space,
        "in_mesh": args.in_mesh,
        "output": args.output,
        "engine": args.engine,
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    out_dir = Path(args.output).parent

    with temp_dir("hollow", keep=args.no_clean, base_dir=out_dir) as tmp_dir:
        L.info("Temporary folder: %s", tmp_dir)

        # ------------------------------------------------------------------ #
        # Load brain mesh
        # ------------------------------------------------------------------ #
        brain_mesh = trimesh.load(args.in_mesh, force="mesh")
        runlog["brain_vertices"] = len(brain_mesh.vertices)
        runlog["steps"].append("Loaded brain mesh")

        # ------------------------------------------------------------------ #
        # Locate aseg & create ventricle binary mask
        # ------------------------------------------------------------------ #
        anat_dir = Path(args.subjects_dir) / args.subject_id / "anat"
        aseg_path = flexible_match(
            base_dir=anat_dir,
            subject_id=args.subject_id,
            descriptor="desc-aseg",
            suffix="dseg",
            session=args.session,
            run=args.run,
            ext=".nii.gz",
        )
        vent_mask_nii = Path(tmp_dir) / f"vent_mask_{args.space}.nii.gz"
        vent_labels = ["4", "5", "14", "15", "43", "44", "72"]

        run_cmd(
            ["mri_binarize", "--i", aseg_path, "--match", *vent_labels, "--o", vent_mask_nii],
            verbose=args.verbose,
        )
        runlog["steps"].append("Created ventricle binary mask")

        # Optional dilation/erosion
        if args.dilate_mask:
            L.info("Dilating ventricle mask …")
            nii = nib.load(vent_mask_nii)
            data = nii.get_fdata() > 0
            struct = generate_binary_structure(3, 2)
            dil = binary_dilation(data, structure=struct, iterations=2)
            filled = binary_erosion(dil, structure=struct, iterations=1)
            vent_mask_nii = Path(tmp_dir) / "vent_mask_filled.nii.gz"
            nib.save(nib.Nifti1Image(filled.astype(np.uint8), nii.affine), vent_mask_nii)
            runlog["steps"].append("Dilated / eroded mask")

        # ------------------------------------------------------------------ #
        # Mask → surface mesh
        # ------------------------------------------------------------------ #
        vent_gii = Path(tmp_dir) / "ventricles.surf.gii"
        volume_to_gifti(str(vent_mask_nii), str(vent_gii), level=0.5)
        vent_mesh = gifti_to_trimesh(str(vent_gii))
        components = vent_mesh.split(only_watertight=False)

        # Filter trivial / non‑volume comps
        kept = []
        for i, comp in enumerate(components):
            comp.remove_degenerate_faces()
            comp.remove_unreferenced_vertices()
            comp.fix_normals()
            if comp.volume > 1.0 and comp.is_volume:
                kept.append(comp)
            else:
                L.warning("Skipping ventricle component %d (volume %.2f)", i, comp.volume)

        if not kept:
            sys.exit("No usable ventricle components found.")

        # ------------------------------------------------------------------ #
        # Boolean subtraction
        # ------------------------------------------------------------------ #
        trimesh.constants.DEFAULT_WEAK_ENGINE = args.engine
        engine = None if args.engine == "auto" else args.engine

        hollow = brain_mesh.copy()
        for comp in kept:
            L.info("Subtracting component (vol %.2f mm³)…", comp.volume)
            try:
                hollow = trimesh.boolean.difference([hollow, comp], engine=engine)
            except Exception as exc:
                L.error("Boolean subtraction failed: %s", exc)
                runlog["warnings"].append(f"Boolean subtraction failed: {exc}")

        if not hollow.is_watertight:
            L.warning("Result not watertight — attempting voxel remesh")
            hollow = voxel_remesh_and_repair(hollow, pitch=0.5, do_smooth=True)

        hollow.export(args.output, file_type="stl")
        runlog["output_files"].append(args.output)
        runlog["steps"].append("Exported hollowed mesh")

    # ---------------- write audit‑log ---------------- #
    write_log(runlog, out_dir, base_name="hollow_log")
    L.info("Done.")


if __name__ == "__main__":
    main()

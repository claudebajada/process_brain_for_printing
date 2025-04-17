#!/usr/bin/env python
# brain_for_printing/cli_slab_slices.py

"""
Slice multiple 3D meshes into volumetric slabs for 3D printing while keeping them
aligned using the same slab intervals. Each slab will have its own consistent
bounding box across all meshes, padded with dummy corner cubes so that each slab
is aligned across all meshes. The bounding box is not global across all slabs.
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import uuid

import numpy as np
import trimesh

from .io_utils import temp_dir, require_cmds
from .mesh_utils import voxel_remesh_and_repair
from .log_utils import get_logger, write_log


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _build_slab_box(bounds, orientation: str, lower: float, upper: float):
    """
    Compute (center, extents) for the oriented bounding box of one slab.
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    if orientation == "axial":  # Z‑axis
        return (
            ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (lower + upper) / 2.0),
            (xmax - xmin, ymax - ymin, upper - lower),
        )
    elif orientation == "coronal":  # Y‑axis
        return (
            ((xmin + xmax) / 2.0, (lower + upper) / 2.0, (zmin + zmax) / 2.0),
            (xmax - xmin, upper - lower, zmax - zmin),
        )
    else:  # sagittal  (X‑axis)
        return (
            ((lower + upper) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0),
            (upper - lower, ymax - ymin, zmax - zmin),
        )


def _pad_with_dummy_cubes(mesh: trimesh.Trimesh, bounds) -> trimesh.Trimesh:
    """
    Add tiny cubes at the eight corners of *bounds* so that when exported all
    meshes share the same bounding box (crucial for alignment).
    """
    min_corner, max_corner = bounds
    corners = np.array(
        np.meshgrid(
            [min_corner[0], max_corner[0]],
            [min_corner[1], max_corner[1]],
            [min_corner[2], max_corner[2]],
        )
    ).T.reshape(-1, 3)

    cubes = [
        trimesh.creation.box(
            extents=[0.001, 0.001, 0.001],
            transform=trimesh.transformations.translation_matrix(c),
        )
        for c in corners
    ]
    return trimesh.util.concatenate([mesh] + cubes)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Slice one or more meshes into volumetric slabs.  All inputs are "
            "cut with identical intervals to guarantee alignment."
        )
    )
    ap.add_argument("--in_meshes", nargs="+", required=True, help="Input mesh files.")
    ap.add_argument(
        "--orientation",
        choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Axis along which to slice.",
    )
    ap.add_argument("--thickness", type=float, default=10.0, help="Slab thickness.")
    ap.add_argument(
        "--engine",
        choices=["scad", "blender", "auto"],
        default="auto",
        help="Boolean backend for trimesh.",
    )
    ap.add_argument("--out_dir", default=".", help="Output directory.")
    ap.add_argument("--no_clean", action="store_true", help="Keep temp folder.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return ap


def main() -> None:
    args = _get_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)

    # optional external tools
    if args.engine in ("scad", "blender"):
        require_cmds([args.engine], logger=L)

    runlog = {
        "tool": "brain_for_printing_slab_slices",
        "in_meshes": args.in_meshes,
        "orientation": args.orientation,
        "thickness": args.thickness,
        "engine": args.engine,
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with temp_dir("slab", keep=args.no_clean, base_dir=out_dir) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        L.info("Temporary folder: %s", tmp_dir)

        # ------------------------------------------------------------------ #
        # Load meshes & compute combined bounds
        # ------------------------------------------------------------------ #
        meshes = []
        overall_min = np.full(3, np.inf)
        overall_max = np.full(3, -np.inf)

        for p in args.in_meshes:
            m = trimesh.load(p, force="mesh")
            if m.is_empty:
                raise ValueError(f"Mesh {p} is empty.")
            meshes.append(m)
            bmin, bmax = m.bounds
            overall_min = np.minimum(overall_min, bmin)
            overall_max = np.maximum(overall_max, bmax)
            runlog["steps"].append(f"Loaded mesh {p}")

        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[args.orientation]
        min_val, max_val = overall_min[axis], overall_max[axis]

        slab_positions = []
        cur = min_val
        while cur < max_val:
            top = min(cur + args.thickness, max_val)
            slab_positions.append((cur, top))
            cur += args.thickness

        if not slab_positions:
            sys.exit("No slabs generated — check thickness or bounds.")

        # dict[ slab_idx ] -> list[(mesh_idx, slab_mesh)]
        slabs: dict[int, list[tuple[int, trimesh.Trimesh]]] = {i: [] for i in range(len(slab_positions))}

        # ------------------------------------------------------------------ #
        # Slice each mesh
        # ------------------------------------------------------------------ #
        trimesh.constants.DEFAULT_WEAK_ENGINE = args.engine
        engine = None if args.engine == "auto" else args.engine

        for midx, mesh in enumerate(meshes):
            if not mesh.is_volume:
                L.warning("Mesh %s not volume — running voxel remesh", args.in_meshes[midx])
                mesh = voxel_remesh_and_repair(mesh, pitch=0.5, do_smooth=True)

            for sidx, (lower, upper) in enumerate(slab_positions):
                center, extents = _build_slab_box(mesh.bounds, args.orientation, lower, upper)
                box = trimesh.creation.box(
                    extents=extents,
                    transform=trimesh.transformations.translation_matrix(center),
                )
                try:
                    slab = trimesh.boolean.intersection([mesh, box], engine=engine)
                except Exception as exc:
                    L.warning("Intersection failed (%s): %s", args.in_meshes[midx], exc)
                    continue
                if slab and not slab.is_empty:
                    if isinstance(slab, trimesh.Scene):
                        geom = [g for g in slab.geometry.values() if g.is_volume]
                        if geom:
                            slab = trimesh.util.concatenate(geom)
                    slabs[sidx].append((midx, slab))

        # ------------------------------------------------------------------ #
        # Export slabs, padding each to its local bounds
        # ------------------------------------------------------------------ #
        for sidx, sublist in slabs.items():
            if not sublist:
                continue

            local_min = np.full(3, np.inf)
            local_max = np.full(3, -np.inf)
            for _, sm in sublist:
                bmin, bmax = sm.bounds
                local_min = np.minimum(local_min, bmin)
                local_max = np.maximum(local_max, bmax)
            local_bounds = (local_min, local_max)

            for midx, sm in sublist:
                padded = _pad_with_dummy_cubes(sm, local_bounds)
                out_path = out_dir / f"mesh{midx}_slab_{sidx:03d}.stl"
                padded.export(out_path)
                runlog["output_files"].append(str(out_path))
                runlog["steps"].append(f"Exported {out_path.name}")

    write_log(runlog, out_dir, base_name="slab_slices_log")
    L.info("Done.")


if __name__ == "__main__":
    main()

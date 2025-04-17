#!/usr/bin/env python

"""
Slice multiple 3D meshes into volumetric slabs for 3D printing while keeping them
aligned using the same slab intervals. Each slab will have its own consistent
bounding box across all meshes, padded with dummy corner cubes so that each slab
is aligned across all meshes. The bounding box is not global across all slabs.
"""

import os
import argparse
import uuid
import shutil
import trimesh
import numpy as np

from .mesh_utils import voxel_remesh_and_repair
from .log_utils import write_log


def pad_with_dummy_cubes(mesh: trimesh.Trimesh, target_bounds: tuple[tuple[float], tuple[float]]) -> trimesh.Trimesh:
    """
    Add tiny cubes at the corners of a target bounding box to ensure exported mesh
    maintains consistent bounding box for that slab.
    """
    min_corner, max_corner = target_bounds

    corners = np.array(
        np.meshgrid(
            [min_corner[0], max_corner[0]],
            [min_corner[1], max_corner[1]],
            [min_corner[2], max_corner[2]]
        )
    ).T.reshape(-1, 3)

    dummy_cubes = []
    for corner in corners:
        # A tiny 1 micrometer cube
        box = trimesh.creation.box(
            extents=[0.001, 0.001, 0.001],
            transform=trimesh.transformations.translation_matrix(corner)
        )
        dummy_cubes.append(box)

    combined = trimesh.util.concatenate([mesh] + dummy_cubes)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Slice multiple 3D mesh files into volumetric slabs for 3D printing. "
            "All input meshes are sliced along identical intervals based on their "
            "combined bounding box in the chosen orientation. Each slab has its own "
            "unified bounding box across all meshes, ensuring alignment among them."
        )
    )
    parser.add_argument("--in_meshes", nargs="+", required=True,
                        help="Paths to one or more input 3D mesh files (STL, OBJ, etc.).")
    parser.add_argument("--orientation", choices=["axial", "coronal", "sagittal"],
                        default="axial",
                        help="Axis to slice along: 'axial' (Z), 'coronal' (Y), or 'sagittal' (X).")
    parser.add_argument("--thickness", type=float, default=10.0,
                        help="Slab thickness (in mesh units, usually mm).")
    parser.add_argument("--engine", choices=["scad", "blender", "auto"],
                        default="auto",
                        help="Boolean engine for trimesh. If 'auto', picks the first available.")
    parser.add_argument("--out_dir", default=".",
                        help="Directory to store the output slab STL files.")
    parser.add_argument("--no_clean", action="store_true",
                        help="If set, do NOT remove the temporary folder at the end.")
    args = parser.parse_args()

    log = {
        "tool": "brain_for_printing_slab_slices",
        "in_meshes": args.in_meshes,
        "orientation": args.orientation,
        "thickness": args.thickness,
        "engine": args.engine,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    # Create a temporary directory for intermediate files
    tmp_dir = os.path.join(args.out_dir, f"_tmp_slab_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    log["steps"].append(f"Created temp dir => {tmp_dir}")

    # ----------------------------------------------------
    # Load all meshes and compute the unified bounding box
    # in the chosen orientation for slicing intervals.
    # ----------------------------------------------------
    all_meshes = []
    overall_min = [float("inf"), float("inf"), float("inf")]  # for orientation-based slicing only
    overall_max = [float("-inf"), float("-inf"), float("-inf")]  # for orientation-based slicing only

    for idx, mesh_path in enumerate(args.in_meshes):
        print(f"[INFO] Loading mesh {idx} => {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh.is_empty:
            raise ValueError(f"Loaded mesh {mesh_path} is empty or invalid.")
        # For slicing intervals, we only care about axis_index min & max.
        bmin, bmax = mesh.bounds
        for d in range(3):
            if bmin[d] < overall_min[d]:
                overall_min[d] = bmin[d]
            if bmax[d] > overall_max[d]:
                overall_max[d] = bmax[d]
        all_meshes.append(mesh)
        log["steps"].append(f"Loaded input mesh {idx} => {mesh_path}")

    if args.orientation == "axial":
        axis_index = 2
    elif args.orientation == "coronal":
        axis_index = 1
    else:  # sagittal
        axis_index = 0

    min_val = overall_min[axis_index]
    max_val = overall_max[axis_index]

    # Build the slab intervals
    slab_positions = []
    current = min_val
    while current < max_val:
        top = current + args.thickness
        if top > max_val:
            top = max_val
        slab_positions.append((current, top))
        current += args.thickness

    if not slab_positions:
        wmsg = "No slabs were computed; check thickness or bounding box."
        print(f"[WARNING] {wmsg}")
        log["warnings"].append(wmsg)

    # For each slab index, store a list of (mesh_idx, slab_mesh)
    slabs_dict = { i: [] for i in range(len(slab_positions)) }

    # ----------------------------------------------------
    # For each mesh, perform the slice boolean intersection
    # We only do a single pass, collecting results in slabs_dict
    # ----------------------------------------------------
    for mesh_idx, mesh in enumerate(all_meshes):
        # Ensure watertight volume
        if not mesh.is_volume:
            msg = f"Mesh {mesh_idx} not a volume. Attempting voxel remesh..."
            print(f"[WARNING] {msg}")
            log["warnings"].append(msg)
            try:
                mesh = voxel_remesh_and_repair(mesh, pitch=0.5, do_smooth=True, smooth_iterations=10)
                if not mesh.is_watertight:
                    raise ValueError("Remeshed mesh is still not watertight!")
            except Exception as e:
                emsg = f"Voxel remesh failed on mesh {mesh_idx}: {e}"
                print(f"[ERROR] {emsg}")
                log["warnings"].append(emsg)
                raise

        for slab_idx, (lower, upper) in enumerate(slab_positions):
            # Create the bounding box for this slab along the chosen orientation
            box_center, box_extents = _build_slab_box(
                mesh.bounds, args.orientation, lower, upper
            )
            box_transform = trimesh.transformations.translation_matrix(box_center)
            box_mesh = trimesh.creation.box(extents=box_extents, transform=box_transform)

            # Perform intersection
            try:
                slab_mesh = trimesh.boolean.intersection(
                    [mesh, box_mesh],
                    engine=args.engine if args.engine != "auto" else None
                )
            except Exception as e:
                print(f"[WARNING] Boolean intersection failed on mesh {mesh_idx}, slab {slab_idx}: {e}")
                log["warnings"].append(
                    f"Boolean intersection failed on mesh {mesh_idx}, slab {slab_idx}: {e}"
                )
                slab_mesh = None

            if not slab_mesh or slab_mesh.is_empty:
                continue
            if isinstance(slab_mesh, trimesh.Scene):
                # Merge geometry into a single mesh
                combined = trimesh.util.concatenate([
                    g for g in slab_mesh.geometry.values() if g.is_volume
                ])
                if not combined or combined.is_empty:
                    continue
                slab_mesh = combined

            # Store
            slabs_dict[slab_idx].append((mesh_idx, slab_mesh))

    # ----------------------------------------------------
    # Now, for each slab_idx, unify bounding boxes across all sub-meshes
    # in that slab, then pad them with dummy cubes, and export.
    # ----------------------------------------------------
    for slab_idx, slab_list in slabs_dict.items():
        if not slab_list:
            continue

        # Compute a local bounding box for this slab across all sub-meshes
        local_min = np.array([float("inf"), float("inf"), float("inf")])
        local_max = np.array([float("-inf"), float("-inf"), float("-inf")])
        for (_, smesh) in slab_list:
            bmin, bmax = smesh.bounds
            local_min = np.minimum(local_min, bmin)
            local_max = np.maximum(local_max, bmax)
        local_bounds = (local_min, local_max)

        # Export each sub-slab, padding to the local bounds
        for (mesh_idx, smesh) in slab_list:
            padded = pad_with_dummy_cubes(smesh, local_bounds)
            out_slab = os.path.join(
                args.out_dir,
                f"mesh{mesh_idx}_slab_{slab_idx:03d}.stl"
            )
            padded.export(out_slab)
            log["steps"].append(
                f"Exported slab {slab_idx:03d} for mesh {mesh_idx} => {out_slab}"
            )
            log["output_files"].append(out_slab)

    # ----------------------------------------------------
    # Write the process log to disk
    # ----------------------------------------------------
    write_log(log, output_dir=args.out_dir, base_name="slab_slices_log")

    # Cleanup
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log["steps"].append(f"Removed temp dir => {tmp_dir}")


def _build_slab_box(original_bounds, orientation, lower, upper):
    """
    Create a bounding box for the slab along the chosen orientation, ignoring other axes.
    We'll slice within the mesh's own bounding box but use the 'lower' and 'upper' on that axis.

    original_bounds: (bmin, bmax) for the mesh.
    orientation: 'axial', 'coronal', or 'sagittal'.
    lower, upper: slab positions on the relevant axis.

    returns (box_center, box_extents)
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = original_bounds

    if orientation == "axial":  # slicing along Z
        x_size = xmax - xmin
        y_size = ymax - ymin
        z_size = upper - lower
        center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (lower + upper) / 2.0
        )
        extents = (x_size, y_size, z_size)
    elif orientation == "coronal":  # slicing along Y
        x_size = xmax - xmin
        y_size = upper - lower
        z_size = zmax - zmin
        center = (
            (xmin + xmax) / 2.0,
            (lower + upper) / 2.0,
            (zmin + zmax) / 2.0
        )
        extents = (x_size, y_size, z_size)
    else:  # sagittal
        x_size = upper - lower
        y_size = ymax - ymin
        z_size = zmax - zmin
        center = (
            (lower + upper) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0
        )
        extents = (x_size, y_size, z_size)

    return center, extents


if __name__ == "__main__":
    main()


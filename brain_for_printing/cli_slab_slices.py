#!/usr/bin/env python

"""
Slice multiple 3D meshes into volumetric slabs for 3D printing while keeping them 
aligned using the same slab intervals. This version omits any subdivision steps 
so that the boolean intersections are simpler to compute. You can create denser 
versions of the slabs later if needed.

Usage Example:
  brain_for_printing_slab_slices \
    --in_meshes brain1.stl brain2.stl \
    --orientation axial \
    --thickness 10 \
    --out_dir slabs_out

All input meshes will be sliced along the same intervals determined by the unified
bounding box of all meshes.
"""

import os
import argparse
import uuid
import shutil
import trimesh

from .mesh_utils import voxel_remesh_and_repair
from .log_utils import write_log


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Slice multiple 3D mesh files into volumetric slabs for 3D printing. "
            "All input meshes are sliced along identical intervals based on their "
            "combined bounding box, without any subdivision step."
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

    # Log dictionary to record process steps and outputs
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
    print(f"[INFO] Temporary folder => {tmp_dir}")
    log["steps"].append(f"Created temp dir => {tmp_dir}")

    # ----------------------------------------------------
    # Load all meshes and compute the unified bounding box
    # ----------------------------------------------------
    all_meshes = []
    overall_min = [float("inf"), float("inf"), float("inf")]
    overall_max = [float("-inf"), float("-inf"), float("-inf")]

    for idx, mesh_path in enumerate(args.in_meshes):
        print(f"[INFO] Loading mesh {idx} => {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh.is_empty:
            raise ValueError(f"Loaded mesh {mesh_path} is empty or invalid.")
        
        # Update the overall bounding box using each mesh's bounds
        bmin, bmax = mesh.bounds
        overall_min = [min(overall_min[d], bmin[d]) for d in range(3)]
        overall_max = [max(overall_max[d], bmax[d]) for d in range(3)]
        
        all_meshes.append(mesh)
        log["steps"].append(f"Loaded input mesh {idx} => {mesh_path}")
        log[f"input_{idx}_vertices"] = len(mesh.vertices)

    # ----------------------------------------------------
    # Determine orientation axis and compute slab intervals
    # from the unified bounding box
    # ----------------------------------------------------
    if args.orientation == "axial":
        axis_index = 2
    elif args.orientation == "coronal":
        axis_index = 1
    else:  # sagittal
        axis_index = 0

    min_val = overall_min[axis_index]
    max_val = overall_max[axis_index]

    # Build slab intervals (each as a tuple: (lower, upper))
    slab_positions = []
    current = min_val
    while current < max_val:
        top = current + args.thickness
        if top > max_val:
            top = max_val
        slab_positions.append((current, top))
        current += args.thickness

    if not slab_positions:
        wmsg = "No slabs were created; check thickness or bounding box."
        print(f"[WARNING] {wmsg}")
        log["warnings"].append(wmsg)

    # ----------------------------------------------------
    # Process each mesh: ensure volume integrity, then slice
    # ----------------------------------------------------
    for mesh_idx, mesh in enumerate(all_meshes):
        if not mesh.is_volume:
            msg = f"Input mesh {mesh_idx} is not a volume. Attempting voxel remeshing..."
            print(f"[WARNING] {msg}")
            log["warnings"].append(msg)
            try:
                mesh = voxel_remesh_and_repair(
                    mesh,
                    pitch=0.5,
                    do_smooth=True,
                    smooth_iterations=10
                )
                if not mesh.is_watertight:
                    raise ValueError(f"Voxel remeshed mesh {mesh_idx} is still not watertight!")
                log["steps"].append(f"Voxel remeshing successful for mesh {mesh_idx}")
            except Exception as e:
                msg = f"Voxel remeshing failed for mesh {mesh_idx}: {e}"
                print(f"[ERROR] {msg}")
                log["warnings"].append(msg)
                raise RuntimeError(msg)
        
        # Slice the current mesh using the shared slab intervals
        slabs_for_this_mesh = []
        for i, (lower, upper) in enumerate(slab_positions):
            # Build a bounding box for the current slab using the overall bounds
            box_center, box_extents = _build_slab_box(
                mesh.bounds, args.orientation, lower, upper,
                override_bounds=(overall_min, overall_max)
            )
            box_transform = trimesh.transformations.translation_matrix(box_center)
            box_mesh = trimesh.creation.box(extents=box_extents, transform=box_transform)

            # Compute the boolean intersection of the mesh and the slab box
            try:
                slab_mesh = trimesh.boolean.intersection(
                    [mesh, box_mesh],
                    engine=args.engine if args.engine != "auto" else None
                )
            except Exception:
                slab_mesh = None

            if not slab_mesh or slab_mesh.is_empty:
                continue

            # If the result is a Scene, merge the geometries into a single mesh
            if isinstance(slab_mesh, trimesh.Scene):
                combined = trimesh.util.concatenate(
                    [g for g in slab_mesh.geometry.values() if g.is_volume]
                )
                if not combined or combined.is_empty:
                    continue
                slab_mesh = combined

            slabs_for_this_mesh.append((i, slab_mesh))
        
        # ----------------------------------------------------
        # Export slabs for the current mesh
        # ----------------------------------------------------
        for (slab_idx, slab_mesh) in slabs_for_this_mesh:
            out_slab = os.path.join(
                args.out_dir,
                f"mesh{mesh_idx}_slab_{slab_idx:03d}.stl"
            )
            slab_mesh.export(out_slab)
            print(f"[INFO] Exported => {out_slab}")
            log["steps"].append(
                f"Exported slab {slab_idx:03d} for mesh {mesh_idx} => {out_slab}"
            )
            log["output_files"].append(out_slab)

    # ----------------------------------------------------
    # Write the process log to disk
    # ----------------------------------------------------
    write_log(log, output_dir=args.out_dir, base_name="slab_slices_log")

    # Cleanup temporary folder if not preserved by option
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log["steps"].append(f"Removed temp dir => {tmp_dir}")
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")


def _build_slab_box(original_bounds, orientation, lower, upper, override_bounds=None):
    """
    Helper to compute the bounding box for a slab defined by lower and upper
    positions along the slicing axis. If 'override_bounds' is provided as a tuple
    (overall_min, overall_max), that range is used for the non-slicing axes.
    """
    if override_bounds is not None:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = override_bounds
    else:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = original_bounds

    if orientation == "axial":  # slicing along Z
        x_size = xmax - xmin
        y_size = ymax - ymin
        z_size = upper - lower
        box_center = ((xmin + xmax) / 2.0,
                      (ymin + ymax) / 2.0,
                      (lower + upper) / 2.0)
        box_extents = (x_size, y_size, z_size)
    elif orientation == "coronal":  # slicing along Y
        x_size = xmax - xmin
        y_size = upper - lower
        z_size = zmax - zmin
        box_center = ((xmin + xmax) / 2.0,
                      (lower + upper) / 2.0,
                      (zmin + zmax) / 2.0)
        box_extents = (x_size, y_size, z_size)
    else:  # sagittal (slicing along X)
        x_size = upper - lower
        y_size = ymax - ymin
        z_size = zmax - zmin
        box_center = ((lower + upper) / 2.0,
                      (ymin + ymax) / 2.0,
                      (zmin + zmax) / 2.0)
        box_extents = (x_size, y_size, z_size)
    return box_center, box_extents


if __name__ == "__main__":
    main()


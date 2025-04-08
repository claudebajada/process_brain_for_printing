#!/usr/bin/env python

"""
Slice multiple 3D meshes into volumetric slabs for 3D printing, keeping them
aligned/in-sync along the same slab intervals. Optionally subdivide the resulting
slabs to increase vertex density.

Usage Example:
  brain_for_printing_slab_slices \
    --in_meshes brain1.stl brain2.stl \
    --orientation axial \
    --thickness 10 \
    --subdivide_max_edge 2.0 \
    --out_dir slabs_out

In this case, 'brain1.stl' and 'brain2.stl' will be sliced along the same
orientation and thickness intervals. The final slices are exported to 'slabs_out/',
labeled by which input mesh and which slab index.
"""

import os
import argparse
import uuid
import shutil
import trimesh

from .mesh_utils import voxel_remesh_and_repair, slice_mesh_into_slabs
from .log_utils import write_log


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Slice multiple 3D mesh files into volumetric slabs for 3D printing. "
            "All input meshes use the same slab intervals, so slices align across them."
        )
    )
    parser.add_argument("--in_meshes", nargs="+", required=True,
                        help="Paths to one or more input 3D mesh files (STL, OBJ, etc.).")
    parser.add_argument("--orientation", choices=["axial", "coronal", "sagittal"],
                        default="axial",
                        help="Which axis to slice along: 'axial' (Z), 'coronal' (Y), or 'sagittal' (X).")
    parser.add_argument("--thickness", type=float, default=10.0,
                        help="Slab thickness in mesh units (usually mm).")
    parser.add_argument("--engine", choices=["scad", "blender", "auto"],
                        default="auto",
                        help="Boolean engine for trimesh. If 'auto', picks the first available.")
    parser.add_argument("--subdivide_max_edge", type=float, default=None,
                        help="If set, subdivide each slab so no edge exceeds this length.")
    parser.add_argument("--out_dir", default=".",
                        help="Where to store the slab .stl files.")
    parser.add_argument("--no_clean", action="store_true",
                        help="If set, do NOT remove the temporary folder at the end.")
    args = parser.parse_args()

    # Prepare a log dictionary
    log = {
        "tool": "brain_for_printing_slab_slices",
        "in_meshes": args.in_meshes,
        "orientation": args.orientation,
        "thickness": args.thickness,
        "engine": args.engine,
        "subdivide_max_edge": args.subdivide_max_edge,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    # Create a temp directory for intermediate files
    tmp_dir = os.path.join(args.out_dir, f"_tmp_slab_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[INFO] Temporary folder => {tmp_dir}")
    log["steps"].append(f"Created temp dir => {tmp_dir}")

    # ----------------------------------------------------
    # Load all meshes and unify bounding box
    # ----------------------------------------------------
    all_meshes = []
    overall_min = [float("inf"), float("inf"), float("inf")]
    overall_max = [float("-inf"), float("-inf"), float("-inf")]

    for idx, mesh_path in enumerate(args.in_meshes):
        print(f"[INFO] Loading mesh {idx} => {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh.is_empty:
            raise ValueError(f"Loaded mesh {mesh_path} is empty or invalid.")

        # Expand the overall bounding box
        bmin, bmax = mesh.bounds
        overall_min = [min(overall_min[d], bmin[d]) for d in range(3)]
        overall_max = [max(overall_max[d], bmax[d]) for d in range(3)]

        all_meshes.append(mesh)
        log["steps"].append(f"Loaded input mesh {idx} => {mesh_path}")
        log[f"input_{idx}_vertices"] = len(mesh.vertices)

    # ----------------------------------------------------
    # Determine orientation axis and slice intervals
    # using the unified bounding box
    # ----------------------------------------------------
    if args.orientation == "axial":
        axis_index = 2
    elif args.orientation == "coronal":
        axis_index = 1
    else:  # sagittal
        axis_index = 0

    min_val = overall_min[axis_index]
    max_val = overall_max[axis_index]

    # Build a list of slab (lower, upper) intervals
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
    # For each mesh, if not watertight, voxel remesh
    # then slice using the same slab intervals
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
                log["steps"].append(f"Voxel remeshing + repair successful for mesh {mesh_idx}")
            except Exception as e:
                msg = f"Voxel remeshing failed for mesh {mesh_idx}: {e}"
                print(f"[ERROR] {msg}")
                log["warnings"].append(msg)
                raise RuntimeError(msg)

        # Now slice this mesh using the shared slab_positions
        # We call the same logic as slice_mesh_into_slabs, but we override bounding
        # box with our known intervals. We'll replicate the box steps inline.
        slabs_for_this_mesh = []
        for i, (lower, upper) in enumerate(slab_positions):
            # Build the bounding box for that slab
            box_center, box_extents = _build_slab_box(
                mesh.bounds, args.orientation, lower, upper,
                override_bounds=(overall_min, overall_max)
            )
            box_transform = trimesh.transformations.translation_matrix(box_center)
            box_mesh = trimesh.creation.box(extents=box_extents, transform=box_transform)

            # Intersection
            try:
                slab_mesh = trimesh.boolean.intersection([mesh, box_mesh],
                                                         engine=args.engine if args.engine != "auto" else None)
            except Exception:
                # In case intersection fails for any reason
                slab_mesh = None

            if not slab_mesh or slab_mesh.is_empty:
                continue

            # If it's a Scene, merge geometries
            if isinstance(slab_mesh, trimesh.Scene):
                combined = trimesh.util.concatenate(
                    [g for g in slab_mesh.geometry.values() if g.is_volume]
                )
                if not combined or combined.is_empty:
                    continue
                slab_mesh = combined

            # Optionally subdivide
            if args.subdivide_max_edge and args.subdivide_max_edge > 0:
                from trimesh.remesh import subdivide_to_size
                v_sub, f_sub = subdivide_to_size(
                    slab_mesh.vertices,
                    slab_mesh.faces,
                    max_edge=args.subdivide_max_edge
                )
                slab_mesh = trimesh.Trimesh(vertices=v_sub, faces=f_sub)

            slabs_for_this_mesh.append((i, slab_mesh))

        # ----------------------------------------------------
        # Export the slabs for this mesh
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
    # Write the log to disk
    # ----------------------------------------------------
    write_log(log, output_dir=args.out_dir, base_name="slab_slices_log")

    # Cleanup
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log["steps"].append(f"Removed temp dir => {tmp_dir}")
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")


def _build_slab_box(original_bounds, orientation, lower, upper, override_bounds=None):
    """
    Helper to build a bounding box (center + extents) for the specified slab
    based on orientation and either the mesh bounding coords or an override.

    If 'override_bounds' is not None, it should be (overall_min, overall_max)
    for all meshes combined. Then we keep the other two axes the same as
    that overall bounding range, while the selected orientation axis is
    from 'lower' to 'upper'.
    """
    if override_bounds is not None:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = override_bounds
    else:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = original_bounds

    if orientation == "axial":  # Z
        x_size = xmax - xmin
        y_size = ymax - ymin
        z_size = upper - lower
        box_center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (lower + upper) / 2.0
        )
        box_extents = (x_size, y_size, z_size)
    elif orientation == "coronal":  # Y
        x_size = xmax - xmin
        y_size = upper - lower
        z_size = zmax - zmin
        box_center = (
            (xmin + xmax) / 2.0,
            (lower + upper) / 2.0,
            (zmin + zmax) / 2.0
        )
        box_extents = (x_size, y_size, z_size)
    else:  # sagittal => X
        x_size = upper - lower
        y_size = ymax - ymin
        z_size = zmax - zmin
        box_center = (
            (lower + upper) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0
        )
        box_extents = (x_size, y_size, z_size)
    return box_center, box_extents


if __name__ == "__main__":
    main()


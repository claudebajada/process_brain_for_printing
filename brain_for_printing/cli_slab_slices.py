# brain_for_printing/cli_slab_slices.py

import os
import argparse
import uuid
import shutil
import numpy as np
import trimesh

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Slice a 3D mesh into volumetric slabs for 3D printing. Optionally subdivide "
            "the resulting slabs to increase vertex density (useful if you plan to color them)."
        )
    )
    parser.add_argument("--in_mesh", required=True,
        help="Path to the input 3D mesh file (STL, OBJ, etc.).")
    parser.add_argument("--orientation", choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Which axis to slice along: "
             "'axial' (Z), 'coronal' (Y), or 'sagittal' (X).")
    parser.add_argument("--thickness", type=float, default=10.0,
        help="Slab thickness in mesh units (usually mm).")
    parser.add_argument("--engine", choices=["scad", "blender", "auto"],
        default="auto",
        help="Boolean engine for trimesh. If 'auto', it will pick the first available.")
    parser.add_argument("--subdivide_max_edge", type=float, default=None,
        help="If set, subdivide each slab so no edge exceeds this length. "
             "E.g., 2.0 => subdivide until all edges < 2 mm.")
    parser.add_argument("--out_dir", default=".",
        help="Where to store the slab .stl files.")
    parser.add_argument("--no_clean", action="store_true",
        help="If set, do NOT remove the temporary folder at the end.")
    args = parser.parse_args()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  A) Setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    tmp_dir = os.path.join(args.out_dir, f"_tmp_slab_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[INFO] Temporary folder => {tmp_dir}")

    # Load the mesh
    print(f"[INFO] Loading mesh => {args.in_mesh}")
    mesh = trimesh.load(args.in_mesh, force='mesh')
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  B) Determine slice axis
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    if args.orientation == "axial":
        axis_index = 2  # Z axis
        min_val, max_val = zmin, zmax
    elif args.orientation == "coronal":
        axis_index = 1  # Y axis
        min_val, max_val = ymin, ymax
    else:
        axis_index = 0  # X axis
        min_val, max_val = xmin, xmax

    # We'll create multiple slabs from min_val to max_val in increments of thickness
    slab_positions = []
    current = min_val
    while current < max_val:
        top = current + args.thickness
        if top > max_val:
            top = max_val
        slab_positions.append((current, top))
        current += args.thickness

    if not slab_positions:
        print("[WARNING] No slabs created; check thickness or bounding box.")
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  C) Slice into slabs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We'll do a boolean intersection of the mesh with a bounding box
    # for each slab sub-range.
    x_min, x_max = xmin, xmax
    y_min, y_max = ymin, ymax
    z_min, z_max = zmin, zmax

    # Set boolean engine if specified
    if args.engine != "auto":
        trimesh.constants.DEFAULT_WEAK_ENGINE = args.engine

    from trimesh.remesh import subdivide_to_size

    for i, (lower, upper) in enumerate(slab_positions):
        # 1) Construct bounding box for the slab
        if args.orientation == "axial":
            # Slicing in Z => bounding box fully covers X,Y; partial in Z
            x_size = x_max - x_min
            y_size = y_max - y_min
            z_size = upper - lower
            box_center = (
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                (lower + upper) / 2.0
            )
            box_extents = (x_size, y_size, z_size)

        elif args.orientation == "coronal":
            # Slicing in Y => bounding box fully covers X,Z; partial in Y
            x_size = x_max - x_min
            y_size = upper - lower
            z_size = z_max - z_min
            box_center = (
                (x_min + x_max) / 2.0,
                (lower + upper) / 2.0,
                (z_min + z_max) / 2.0
            )
            box_extents = (x_size, y_size, z_size)

        else:  # sagittal
            # Slicing in X => bounding box fully covers Y,Z; partial in X
            x_size = upper - lower
            y_size = y_max - y_min
            z_size = z_max - z_min
            box_center = (
                (lower + upper) / 2.0,
                (y_min + y_max) / 2.0,
                (z_min + z_max) / 2.0
            )
            box_extents = (x_size, y_size, z_size)

        box_transform = trimesh.transformations.translation_matrix(box_center)
        box_mesh = trimesh.creation.box(extents=box_extents, transform=box_transform)

        # 2) Boolean intersection => slab mesh
        slab_mesh = trimesh.boolean.intersection([mesh, box_mesh])
        if not slab_mesh or slab_mesh.is_empty:
            print(f"[INFO] Slab {i} is empty (no intersection). Skipping.")
            continue

        # If the intersection is a Scene with multiple geometry, combine them
        if isinstance(slab_mesh, trimesh.Scene):
            combined = trimesh.util.concatenate(
                [g for g in slab_mesh.geometry.values() if g.is_volume]
            )
            if not combined or combined.is_empty:
                print(f"[INFO] Slab {i} is empty after combine. Skipping.")
                continue
            slab_mesh = combined

        # 3) (Optional) Subdivide to increase vertex density
        if args.subdivide_max_edge and args.subdivide_max_edge > 0:
            # Subdivide until no edge is longer than subdivide_max_edge
            # This can greatly increase the number of faces!
            print(f"[INFO] Subdividing slab {i}: max_edge={args.subdivide_max_edge}")
            v_sub, f_sub = subdivide_to_size(
                slab_mesh.vertices,      # old: slab_mesh.vertices
                slab_mesh.faces,         # old: slab_mesh.faces
                max_edge=args.subdivide_max_edge
            )
            slab_mesh = trimesh.Trimesh(vertices=v_sub, faces=f_sub)


        # 4) Save
        out_slab = os.path.join(args.out_dir, f"slab_{i:03d}.stl")
        slab_mesh.export(out_slab)
        print(f"[INFO] Exported => {out_slab}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  D) Cleanup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")


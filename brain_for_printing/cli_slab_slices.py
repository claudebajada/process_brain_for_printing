# brain_for_printing/cli_slab_slices.py

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
            "Slice a 3D mesh into volumetric slabs for 3D printing. Optionally subdivide "
            "the resulting slabs to increase vertex density (useful if you plan to color them)."
        )
    )
    parser.add_argument("--in_mesh", required=True,
        help="Path to the input 3D mesh file (STL, OBJ, etc.).")
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
        "in_mesh": args.in_mesh,
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

    # Load the mesh
    print(f"[INFO] Loading mesh => {args.in_mesh}")
    mesh = trimesh.load(args.in_mesh, force='mesh')
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty or invalid.")
    log["steps"].append("Loaded input mesh")
    log["input_vertices"] = len(mesh.vertices)

    # If mesh isn't a volume, attempt voxel-based remesh
    if not mesh.is_volume:
        msg = "Input mesh is not a volume. Attempting voxel remeshing..."
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
                raise ValueError("Voxel remeshed mesh is still not watertight!")
            log["steps"].append("Voxel remeshing + repair successful")
        except Exception as e:
            msg = f"Voxel remeshing failed: {e}"
            print(f"[ERROR] {msg}")
            log["warnings"].append(msg)
            # Raise an error to stop or handle gracefully:
            raise RuntimeError(msg)

    # Slice the mesh
    slabs = slice_mesh_into_slabs(
        mesh,
        orientation=args.orientation,
        thickness=args.thickness,
        subdivide_max_edge=args.subdivide_max_edge,
        engine=args.engine
    )
    if not slabs:
        wmsg = "No slabs were created; check thickness or bounding box."
        print(f"[WARNING] {wmsg}")
        log["warnings"].append(wmsg)
        # Optionally, we can return or exit here
        # return

    # Export each slab
    for i, slab_mesh in enumerate(slabs):
        out_slab = os.path.join(args.out_dir, f"slab_{i:03d}.stl")
        slab_mesh.export(out_slab)
        print(f"[INFO] Exported => {out_slab}")
        log["steps"].append(f"Exported slab {i:03d} => {out_slab}")
        log["output_files"].append(out_slab)

    # Write the log to disk
    write_log(log, output_dir=args.out_dir, base_name="slab_slices_log")

    # Cleanup
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log["steps"].append(f"Removed temp dir => {tmp_dir}")
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")

if __name__ == "__main__":
    main()


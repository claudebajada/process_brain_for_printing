# brain_for_printing/cli_color.py

import os
import argparse
import uuid
import shutil
import trimesh

from .color_utils import project_param_to_surface
from .mesh_utils import gifti_to_trimesh

def main():
    parser = argparse.ArgumentParser(
        description="Color an existing GIFTI/OBJ mesh using a param map."
    )
    parser.add_argument("--in_mesh", required=True,
        help="Path to an existing mesh (GIFTI, OBJ, STL, etc.).")
    parser.add_argument("--param_map", required=True,
        help="Path to a param volume (NIfTI) to be sampled.")
    parser.add_argument("--out_obj", default="colored_mesh.obj",
        help="Output OBJ file name.")
    parser.add_argument("--num_colors", type=int, default=6)
    parser.add_argument("--no_clean", action="store_true",
        help="If set, do NOT remove the temporary folder.")
    args = parser.parse_args()

    tmp_dir = f"_tmp_color_{uuid.uuid4().hex[:6]}"
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[INFO] Temp folder => {tmp_dir}")

    # Load
    mesh = None
    # trimesh can load many file types automatically, but GIFTI requires nibabel approach
    # so let's see if it's a .gii
    if args.in_mesh.endswith(".gii"):
        mesh = gifti_to_trimesh(args.in_mesh)
    else:
        mesh = trimesh.load(args.in_mesh)

    # Color
    colored = project_param_to_surface(
        mesh, 
        args.param_map, 
        num_colors=args.num_colors
    )

    # Export
    colored.export(args.out_obj, file_type="obj")
    print(f"[INFO] Colored OBJ => {args.out_obj}")

    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[INFO] Removed temp folder => {tmp_dir}")
    else:
        print(f"[INFO] Temp folder retained => {tmp_dir}")


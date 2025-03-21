# brain_for_printing/cli_t1.py

import os
import argparse
import uuid
import shutil
import trimesh

from .io_utils import first_match
from .mesh_utils import gifti_to_trimesh
from .surfaces import extract_brainstem_in_t1
from .color_utils import (
    project_param_to_surface,
    color_pial_from_midthickness
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate T1-space surfaces for a given subject "
                    "(LH/RH pial or white, optional brainstem), optionally color them."
    )
    parser.add_argument("--subjects_dir", required=True,
        help="Path to derivatives or subject data.")
    parser.add_argument("--subject_id", required=True,
        help="Subject identifier matching your derivatives naming.")
    parser.add_argument("--output_dir", default=".",
        help="Where to store the output files (STL, OBJ, etc.).")
    parser.add_argument("--no_brainstem", action="store_true",
        help="If set, skip extracting the brainstem.")
    parser.add_argument("--no_fill", action="store_true",
        help="Skip hole-filling in the extracted brainstem mesh.")
    parser.add_argument("--no_smooth", action="store_true",
        help="Skip Taubin smoothing in the extracted brainstem mesh.")
    parser.add_argument("--param_map", default=None,
        help="Path to a param volume in T1 space.")
    parser.add_argument("--use_midthickness", action="store_true",
        help="If set, param_map is sampled on the mid surface and copied to pial/white.")
    parser.add_argument("--use_white", action="store_true",
        help="Use white matter surfaces instead of pial surfaces.")
    parser.add_argument("--num_colors", type=int, default=6,
        help="Number of discrete color steps if param_map is provided.")
    parser.add_argument("--export_obj", action="store_true",
        help="Export colored OBJ if param_map is provided.")
    parser.add_argument("--no_clean", action="store_true",
        help="If set, do NOT remove the temporary folder at the end.")
    args = parser.parse_args()

    do_fill = not args.no_fill
    do_smooth = not args.no_smooth
    do_clean = not args.no_clean

    tmp_dir = os.path.join(args.output_dir, f"_tmp_t1_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[INFO] Temporary folder => {tmp_dir}")

    anat_dir = os.path.join(args.subjects_dir, args.subject_id, "anat")

    # Determine which surface type to load
    surf_type = "white" if args.use_white else "pial"

    # Identify surface files
    lh_surf_pattern = f"{anat_dir}/*_run-01_hemi-L_{surf_type}.surf.gii"
    rh_surf_pattern = f"{anat_dir}/*_run-01_hemi-R_{surf_type}.surf.gii"
    lh_surf_file = first_match(lh_surf_pattern)
    rh_surf_file = first_match(rh_surf_pattern)

    # Optionally get mid surfaces if coloring that way
    lh_mid_file = rh_mid_file = None
    if args.use_midthickness:
        lh_mid_pattern = f"{anat_dir}/*_run-01_hemi-L_midthickness.surf.gii"
        rh_mid_pattern = f"{anat_dir}/*_run-01_hemi-R_midthickness.surf.gii"
        lh_mid_file = first_match(lh_mid_pattern)
        rh_mid_file = first_match(rh_mid_pattern)

    # Load cortical surfaces
    lh_mesh_t1 = gifti_to_trimesh(lh_surf_file)
    rh_mesh_t1 = gifti_to_trimesh(rh_surf_file)

    # Extract brainstem if requested
    st_mesh_t1 = None
    if not args.no_brainstem:
        bs_t1_gii = extract_brainstem_in_t1(
            subjects_dir=args.subjects_dir,
            subject_id=args.subject_id,
            tmp_dir=tmp_dir
        )
        st_mesh_t1 = gifti_to_trimesh(bs_t1_gii)
        if do_fill:
            trimesh.repair.fill_holes(st_mesh_t1)
        if do_smooth:
            trimesh.smoothing.filter_taubin(st_mesh_t1, lamb=0.5, nu=-0.53, iterations=10)
        st_mesh_t1.invert()

    # Combine cortical + brainstem
    if st_mesh_t1:
        t1_mesh_final = lh_mesh_t1 + rh_mesh_t1 + st_mesh_t1
    else:
        t1_mesh_final = lh_mesh_t1 + rh_mesh_t1

    out_stl = os.path.join(args.output_dir, f"{args.subject_id}_T1_brain.stl")
    t1_mesh_final.export(out_stl, file_type="stl")
    print(f"[INFO] Exported => {out_stl}")

    # Optional param map coloring
    if args.param_map:
        if args.use_midthickness:
            lh_colored = color_pial_from_midthickness(
                lh_surf_file, lh_mid_file, args.param_map,
                num_colors=args.num_colors
            )
            rh_colored = color_pial_from_midthickness(
                rh_surf_file, rh_mid_file, args.param_map,
                num_colors=args.num_colors
            )
        else:
            lh_colored = project_param_to_surface(
                lh_mesh_t1, args.param_map, num_colors=args.num_colors
            )
            rh_colored = project_param_to_surface(
                rh_mesh_t1, args.param_map, num_colors=args.num_colors
            )

        if st_mesh_t1:
            colored_t1 = lh_colored + rh_colored + st_mesh_t1
        else:
            colored_t1 = lh_colored + rh_colored

        if args.export_obj:
            out_obj = os.path.join(args.output_dir, f"{args.subject_id}_T1_colored.obj")
            colored_t1.export(out_obj, file_type="obj")
            print(f"[INFO] Exported => {out_obj}")

    # Cleanup
    if do_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temporary folder retained => {tmp_dir}")

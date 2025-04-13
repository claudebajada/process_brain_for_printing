# brain_for_printing/cli_color.py

import os
import argparse
import uuid
import shutil
import trimesh

from .color_utils import project_param_to_surface
from .log_utils import write_log
from .surfaces import generate_brain_surfaces
from .mesh_utils import gifti_to_trimesh

def main():
    parser = argparse.ArgumentParser(
        description=(
            "A multi-purpose color command with two subcommands:\n"
            "  1) direct  : Directly color an existing mesh with a param map.\n"
            "  2) surface : Generate (pial/white/mid) surfaces in T1/MNI with optional brainstem,\n"
            "               then color them with a param map."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --------------------------------------------------
    # 1) DIRECT SUBCOMMAND
    # --------------------------------------------------
    p_direct = subparsers.add_parser("direct",
        help="Direct param-map coloring of an existing mesh.")
    p_direct.add_argument("--in_mesh", required=True,
        help="Path to a mesh file (STL, OBJ, GIFTI, etc.) to color.")
    p_direct.add_argument("--param_map", required=True,
        help="Param volume (NIfTI) to sample at each vertex.")
    p_direct.add_argument("--param_threshold", type=float, default=None,
        help="Optional threshold. Vertices with param < threshold can be omitted or have special color.")
    p_direct.add_argument("--out_obj", default="colored_mesh.obj",
        help="Output path for the colored OBJ file.")
    p_direct.add_argument("--num_colors", type=int, default=6,
        help="Number of color bins.")
    p_direct.add_argument("--order", type=int, default=1,
        help="Interpolation order (0=nearest, 1=linear, etc.)")
    p_direct.add_argument("--no_clean", action="store_true",
        help="If set, do NOT remove the temporary folder.")

    # --------------------------------------------------
    # 2) SURFACE SUBCOMMAND
    # --------------------------------------------------
    p_surface = subparsers.add_parser("surface",
        help="Generate T1 or MNI surfaces (LH+RH + optional brainstem) of a chosen type, then color them.")
    p_surface.add_argument("--subjects_dir", required=True,
        help="Path to derivatives or subject data.")
    p_surface.add_argument("--subject_id", required=True,
        help="Subject identifier.")
    p_surface.add_argument("--space", choices=["T1","MNI"], default="T1",
        help="Output space: T1 (native) or MNI.")
    p_surface.add_argument("--surf_type", choices=["pial","white","mid"], default="pial",
        help="Which cortical surface to generate: pial, white, or mid.")
    p_surface.add_argument("--output_dir", default=".",
        help="Where to store the output STL files.")
    p_surface.add_argument("--no_brainstem", action="store_true",
        help="Skip extracting the brainstem.")
    p_surface.add_argument("--split_hemis", action="store_true",
        help="Export LH, RH, (and optional brainstem) as separate STL files instead of merging.")
    p_surface.add_argument("--param_map", required=True,
        help="Param volume (NIfTI) used to color the generated surfaces.")
    p_surface.add_argument("--param_threshold", type=float, default=None,
        help="Optional threshold. Vertices with param < threshold can be omitted or have special color.")
    p_surface.add_argument("--num_colors", type=int, default=6,
        help="Number of color bins.")
    p_surface.add_argument("--order", type=int, default=1,
        help="Interpolation order (0=nearest, 1=linear, etc.)")
    p_surface.add_argument("--no_fill", action="store_true",
        help="Skip hole-filling in the brainstem.")
    p_surface.add_argument("--no_smooth", action="store_true",
        help="Skip smoothing in the brainstem.")
    p_surface.add_argument("--out_warp", default="warp.nii",
        help="(MNI only) Name of the 4D warp field to create, if needed.")
    p_surface.add_argument("--run", default=None,
        help="Run ID, e.g., run-01, if your filenames include it.")
    p_surface.add_argument("--session", default=None,
        help="Session ID, e.g., ses-01, if your filenames include it.")
    p_surface.add_argument("--verbose", action="store_true")
    p_surface.add_argument("--no_clean", action="store_true")

    args = parser.parse_args()

    if args.mode == "direct":
        run_direct(args)
    elif args.mode == "surface":
        run_surface(args)
    else:
        parser.print_help()


def run_direct(args):
    """
    Subcommand: Direct param map -> existing mesh
    """
    # Basic checks
    if not os.path.isfile(args.in_mesh):
        raise FileNotFoundError(f"in_mesh not found: {args.in_mesh}")
    if not os.path.isfile(args.param_map):
        raise FileNotFoundError(f"param_map not found: {args.param_map}")

    log = {
        "tool": "brain_for_printing_color_direct",
        "in_mesh": args.in_mesh,
        "param_map": args.param_map,
        "param_threshold": args.param_threshold,
        "out_obj": args.out_obj,
        "num_colors": args.num_colors,
        "order": args.order,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    tmp_dir = f"_tmp_color_{uuid.uuid4().hex[:6]}"
    os.makedirs(tmp_dir, exist_ok=True)
    log["steps"].append(f"Created temp folder => {tmp_dir}")
    print(f"[INFO] Temp folder => {tmp_dir}")

    # Load mesh
    if args.in_mesh.endswith(".gii"):
        main_mesh = gifti_to_trimesh(args.in_mesh)
        log["steps"].append("Loaded GIFTI mesh")
    else:
        main_mesh = trimesh.load(args.in_mesh)
        log["steps"].append("Loaded mesh (non-GIFTI)")

    if main_mesh.is_empty:
        raise ValueError("Mesh is empty or invalid.")

    # Color
    log["steps"].append("Applying direct param sampling")
    colored_mesh = project_param_to_surface(
        mesh=main_mesh,
        param_nifti_path=args.param_map,
        num_colors=args.num_colors,
        order=args.order,
        threshold=args.param_threshold   # <--- pass threshold here
    )

    # Export
    colored_mesh.export(args.out_obj, file_type="obj")
    log["steps"].append(f"Exported => {args.out_obj}")
    log["output_files"].append(args.out_obj)
    print(f"[INFO] Colored OBJ => {args.out_obj}")

    # Write log
    write_log(log, ".", base_name="color_log")

    # Cleanup
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log["steps"].append(f"Removed temp => {tmp_dir}")
    else:
        print(f"[INFO] Temp folder retained => {tmp_dir}")


def run_surface(args):
    """
    Subcommand:
      1) Generate T1 or MNI hemisphere surfaces (LH & RH + optional brainstem) 
         of the given --surf_type (pial, white, or mid).
      2) Color them with the param map.
      3) Merge or split hemispheres, then export STL(s).
    """
    # Basic checks
    if not os.path.isfile(args.param_map):
        raise FileNotFoundError(f"param_map not found: {args.param_map}")

    log = {
        "tool": "brain_for_printing_color_surface",
        "subject_id": args.subject_id,
        "space": args.space,
        "surf_type": args.surf_type,
        "no_brainstem": args.no_brainstem,
        "split_hemis": args.split_hemis,
        "param_map": args.param_map,
        "param_threshold": args.param_threshold,
        "num_colors": args.num_colors,
        "order": args.order,
        "output_dir": args.output_dir,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    tmp_dir = os.path.join(args.output_dir, f"_tmp_surfcolor_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    log["steps"].append(f"Created temp dir => {tmp_dir}")
    if args.verbose:
        print(f"[INFO] Temporary folder => {tmp_dir}")

    do_fill = not args.no_fill
    do_smooth = not args.no_smooth
    do_clean = not args.no_clean

    # Generate the requested cortical surface plus optional brainstem
    from .surfaces import generate_brain_surfaces
    all_meshes = generate_brain_surfaces(
        subjects_dir=args.subjects_dir,
        subject_id=args.subject_id,
        space=args.space,
        surfaces=(args.surf_type,),
        no_brainstem=args.no_brainstem,
        no_fill=not do_fill,
        no_smooth=not do_smooth,
        out_warp=args.out_warp,
        run=args.run,
        session=args.session,
        verbose=args.verbose,
        tmp_dir=tmp_dir
    )
    log["steps"].append(f"Generated {args.surf_type} LH/RH surfaces in {args.space} space")

    lh_mesh = all_meshes[f"{args.surf_type}_L"]
    rh_mesh = all_meshes[f"{args.surf_type}_R"]
    bs_mesh = all_meshes["brainstem"]  # could be None

    # Now color them with param_map
    log["steps"].append("Coloring LH with param map")
    project_param_to_surface(
        mesh=lh_mesh,
        param_nifti_path=args.param_map,
        num_colors=args.num_colors,
        order=args.order,
        threshold=args.param_threshold
    )
    log["steps"].append("Coloring RH with param map")
    project_param_to_surface(
        mesh=rh_mesh,
        param_nifti_path=args.param_map,
        num_colors=args.num_colors,
        order=args.order,
        threshold=args.param_threshold
    )

    if bs_mesh:
        log["steps"].append("Coloring brainstem with param map")
        project_param_to_surface(
            mesh=bs_mesh,
            param_nifti_path=args.param_map,
            num_colors=args.num_colors,
            order=args.order,
            threshold=args.param_threshold
        )

    # Export merged or splitted STLs
    if not args.split_hemis:
        final_mesh = lh_mesh + rh_mesh
        if bs_mesh:
            final_mesh += bs_mesh

        out_stl = os.path.join(args.output_dir,
            f"{args.subject_id}_{args.space}_{args.surf_type}_brain_colored.stl")
        final_mesh.export(out_stl, file_type="stl")
        log["steps"].append(f"Exported merged => {out_stl}")
        log["output_files"].append(out_stl)
    else:
        # Separate export
        lh_out = os.path.join(args.output_dir,
            f"{args.subject_id}_{args.space}_{args.surf_type}_LH_colored.stl")
        rh_out = os.path.join(args.output_dir,
            f"{args.subject_id}_{args.space}_{args.surf_type}_RH_colored.stl")
        lh_mesh.export(lh_out, file_type="stl")
        rh_mesh.export(rh_out, file_type="stl")
        log["steps"].append(f"Exported LH => {lh_out}")
        log["steps"].append(f"Exported RH => {rh_out}")
        log["output_files"].extend([lh_out, rh_out])

        if bs_mesh:
            bs_out = os.path.join(args.output_dir,
                f"{args.subject_id}_{args.space}_brainstem_colored.stl")
            bs_mesh.export(bs_out, file_type="stl")
            log["steps"].append(f"Exported brainstem => {bs_out}")
            log["output_files"].append(bs_out)

    # Write log
    write_log(log, args.output_dir, base_name="surface_color_log")

    # Cleanup
    if do_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if args.verbose:
            print(f"[INFO] Removed temp => {tmp_dir}")
        log["steps"].append(f"Removed temp => {tmp_dir}")
    else:
        if args.verbose:
            print(f"[INFO] Temporary folder retained => {tmp_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# brain_for_printing/cli_color.py
#
# Two‑mode command:
#   1) direct  – color an existing mesh from a param map.
#   2) surface – generate cortical surfaces, then color them.
#
# Uses shared helpers: get_logger(), temp_dir(), write_log().

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys

import trimesh

from .io_utils import temp_dir, require_cmds
from .log_utils import get_logger, write_log
from .color_utils import project_param_to_surface
from .mesh_utils import gifti_to_trimesh
from .surfaces import generate_brain_surfaces


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Color brain meshes either directly or after surface generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- direct ----
    d = sub.add_parser("direct", help="Color an existing mesh.")
    d.add_argument("--in_mesh", required=True)
    d.add_argument("--param_map", required=True)
    d.add_argument("--out_obj", default="colored_mesh.obj")
    d.add_argument("--param_threshold", type=float, default=None)
    d.add_argument("--num_colors", type=int, default=6)
    d.add_argument("--order", type=int, default=1)
    d.add_argument("--no_clean", action="store_true")

    # ---- surface ----
    s = sub.add_parser("surface", help="Generate + color cortical surfaces.")
    s.add_argument("--subjects_dir", required=True)
    s.add_argument("--subject_id", required=True)
    s.add_argument("--space", choices=["T1", "MNI"], default="T1")
    s.add_argument("--surf_type", choices=["pial", "white", "mid"], default="pial")
    s.add_argument("--output_dir", default=".")
    s.add_argument("--no_brainstem", action="store_true")
    s.add_argument("--split_hemis", action="store_true")
    s.add_argument("--param_map", required=True)
    s.add_argument("--param_threshold", type=float, default=None)
    s.add_argument("--num_colors", type=int, default=6)
    s.add_argument("--order", type=int, default=1)
    s.add_argument("--no_fill", action="store_true")
    s.add_argument("--no_smooth", action="store_true")
    s.add_argument("--out_warp", default="warp.nii")
    s.add_argument("--run", default=None)
    s.add_argument("--session", default=None)
    s.add_argument("--no_clean", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")

    # global verbose for both modes
    p.add_argument("-v", "--verbose", action="store_true", help=argparse.SUPPRESS)
    return p


# --------------------------------------------------------------------------- #
# Mode implementations
# --------------------------------------------------------------------------- #
def _run_direct(args, logger) -> None:
    if not Path(args.in_mesh).is_file():
        sys.exit(f"in_mesh not found: {args.in_mesh}")
    if not Path(args.param_map).is_file():
        sys.exit(f"param_map not found: {args.param_map}")

    runlog = {
        "tool": "brain_for_printing_color_direct",
        "in_mesh": args.in_mesh,
        "param_map": args.param_map,
        "param_threshold": args.param_threshold,
        "out_obj": args.out_obj,
        "num_colors": args.num_colors,
        "order": args.order,
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    with temp_dir("color", keep=args.no_clean) as tmp:
        logger.info("Temp folder: %s", tmp)

        mesh = (
            gifti_to_trimesh(args.in_mesh)
            if args.in_mesh.endswith(".gii")
            else trimesh.load(args.in_mesh)
        )
        runlog["steps"].append("Loaded mesh")

        project_param_to_surface(
            mesh=mesh,
            param_nifti_path=args.param_map,
            num_colors=args.num_colors,
            order=args.order,
            threshold=args.param_threshold,
        )
        runlog["steps"].append("Applied param‑map colouring")

        mesh.export(args.out_obj, file_type="obj")
        runlog["steps"].append(f"Exported OBJ ⇒ {args.out_obj}")
        runlog["output_files"].append(args.out_obj)

    write_log(runlog, ".", base_name="color_direct_log")
    logger.info("Done.")


def _run_surface(args, logger) -> None:
    if not Path(args.param_map).is_file():
        sys.exit(f"param_map not found: {args.param_map}")

    runlog = {
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
        "output_files": [],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with temp_dir("surfcolor", keep=args.no_clean, base_dir=out_dir) as tmp:
        logger.info("Temporary folder: %s", tmp)

        # ---------- generate surfaces ----------
        meshes = generate_brain_surfaces(
            subjects_dir=args.subjects_dir,
            subject_id=args.subject_id,
            space=args.space,
            surfaces=(args.surf_type,),
            no_brainstem=args.no_brainstem,
            no_fill=args.no_fill,
            no_smooth=args.no_smooth,
            out_warp=args.out_warp,
            run=args.run,
            session=args.session,
            verbose=args.verbose,
            tmp_dir=tmp,
        )
        runlog["steps"].append("Generated surfaces")

        lh = meshes[f"{args.surf_type}_L"]
        rh = meshes[f"{args.surf_type}_R"]
        bs = meshes["brainstem"]

        # ---------- colour ----------
        for m, tag in [(lh, "LH"), (rh, "RH")]:
            project_param_to_surface(
                mesh=m,
                param_nifti_path=args.param_map,
                num_colors=args.num_colors,
                order=args.order,
                threshold=args.param_threshold,
            )
            runlog["steps"].append(f"Coloured {tag}")

        if bs:
            project_param_to_surface(
                mesh=bs,
                param_nifti_path=args.param_map,
                num_colors=args.num_colors,
                order=args.order,
                threshold=args.param_threshold,
            )
            runlog["steps"].append("Coloured brainstem")

        # ---------- export ----------
        if not args.split_hemis:
            merged = lh + rh + (bs if bs else trimesh.Trimesh())
            out_obj = out_dir / f"{args.subject_id}_{args.space}_{args.surf_type}_brain_colored.obj"
            merged.export(out_obj, file_type="obj")
            runlog["output_files"].append(str(out_obj))
            runlog["steps"].append(f"Exported merged ⇒ {out_obj}") # Update log message

        else:
            lh_out = out_dir / f"{args.subject_id}_{args.space}_{args.surf_type}_LH_colored.obj"
            rh_out = out_dir / f"{args.subject_id}_{args.space}_{args.surf_type}_RH_colored.obj"
            lh.export(lh_out, file_type="obj")
            rh.export(rh_out, file_type="obj")
            runlog["output_files"] += [str(lh_out), str(rh_out)]
            runlog["steps"] += [f"Exported LH ⇒ {lh_out}", f"Exported RH ⇒ {rh_out}"]

            if bs:
                # Change filename extension and file_type for brainstem
                bs_out = out_dir / f"{args.subject_id}_{args.space}_brainstem_colored.obj"
                bs.export(bs_out, file_type="obj")
                runlog["output_files"].append(str(bs_out))
                # Update log message
                runlog["steps"].append(f"Exported brainstem ⇒ {bs_out}")

    write_log(runlog, out_dir, base_name="color_surface_log")
    logger.info("Done.")


# --------------------------------------------------------------------------- #
# Main dispatcher
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _build_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)

    if args.mode == "direct":
        _run_direct(args, L)
    elif args.mode == "surface":
        _run_surface(args, L)
    else:
        sys.exit("Unknown mode")


if __name__ == "__main__":
    main()

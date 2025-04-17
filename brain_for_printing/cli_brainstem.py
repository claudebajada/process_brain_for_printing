#!/usr/bin/env python
# brain_for_printing/cli_brainstem.py
#
# Extract only the brainstem (T1 or MNI) and export as STL.

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import trimesh

from .io_utils import temp_dir
from .log_utils import get_logger, write_log
from .surfaces import extract_brainstem_in_t1, extract_brainstem_in_mni
from .mesh_utils import gifti_to_trimesh


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract brainstem surface (T1 or MNI).")
    p.add_argument("--subjects_dir", required=True)
    p.add_argument("--subject_id", required=True)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--space", choices=["T1", "MNI"], default="T1")
    p.add_argument("--no_fill", action="store_true")
    p.add_argument("--no_smooth", action="store_true")
    p.add_argument("--run", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--no_clean", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runlog = {
        "tool": "brain_for_printing_brainstem",
        "subject_id": args.subject_id,
        "space": args.space,
        "no_fill": args.no_fill,
        "no_smooth": args.no_smooth,
        "output_dir": str(out_dir),
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    with temp_dir("brainstem", keep=args.no_clean, base_dir=out_dir) as tmp:
        L.info("Temporary folder: %s", tmp)

        if args.space.upper() == "T1":
            gii = extract_brainstem_in_t1(
                subjects_dir=args.subjects_dir,
                subject_id=args.subject_id,
                tmp_dir=tmp,
                verbose=args.verbose,
                session=args.session,
                run=args.run,
            )
        else:
            aseg_mni = Path(tmp) / "aseg_in_mni.nii.gz"
            gii = extract_brainstem_in_mni(
                subjects_dir=args.subjects_dir,
                subject_id=args.subject_id,
                out_aseg_in_mni=aseg_mni,
                tmp_dir=tmp,
                verbose=args.verbose,
                session=args.session,
                run=args.run,
            )
        runlog["steps"].append("Extracted brainstem mask & surface")

        mesh = gifti_to_trimesh(gii)
        if not args.no_fill:
            trimesh.repair.fill_holes(mesh)
            runlog["steps"].append("Filled holes")
        if not args.no_smooth:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
            runlog["steps"].append("Smoothed mesh")

        mesh.invert()
        runlog["steps"].append("Inverted normals")

        out_stl = out_dir / f"{args.subject_id}_brainstem_{args.space}.stl"
        mesh.export(out_stl, file_type="stl")
        runlog["output_files"].append(str(out_stl))
        runlog["steps"].append(f"Exported STL ⇒ {out_stl}")

    write_log(runlog, out_dir, base_name="brainstem_log")
    L.info("Done.")


if __name__ == "__main__":
    main()

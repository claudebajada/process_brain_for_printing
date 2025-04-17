#!/usr/bin/env python
# brain_for_printing/cli_cortical_surfaces.py
#
# Generate LH / RH cortical surfaces (pial, mid, or white) in T1 or MNI space,
# optionally add the brainstem, then export as STL — either merged or split.
# Uses the shared helpers: get_logger(), temp_dir(), require_cmd(), write_log().

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Tuple

import trimesh

from .io_utils import temp_dir, require_cmds
from .log_utils import get_logger, write_log
from .surfaces import generate_brain_surfaces


# --------------------------------------------------------------------------- #
# CLI argument parser
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate cortical surfaces (LH + RH, optional brainstem) in either "
            "T1 or MNI space and export them as STL.  No colouring is done here."
        )
    )
    ap.add_argument("--subjects_dir", required=True, help="BIDS derivatives root.")
    ap.add_argument("--subject_id", required=True, help="e.g. sub‑01")
    ap.add_argument("--space", choices=["T1", "MNI"], default="T1")
    ap.add_argument(
        "--surf_type",
        choices=["pial", "white", "mid"],
        default="pial",
        help="Surface type to export.",
    )
    ap.add_argument("--output_dir", default=".")
    ap.add_argument("--no_brainstem", action="store_true")
    ap.add_argument("--no_fill", action="store_true", help="Skip hole‑filling BS mesh.")
    ap.add_argument("--no_smooth", action="store_true", help="Skip smoothing BS mesh.")
    ap.add_argument(
        "--split_hemis",
        action="store_true",
        help="Export LH / RH / BS separately instead of a single merged STL.",
    )
    ap.add_argument("--out_warp", default="warp.nii", help="Filename for 4‑D warp.")
    ap.add_argument("--run", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--no_clean", action="store_true", help="Keep temp folder.")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print INFO‑level messages and external‑command output.",
    )
    return ap


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _export_split(
    lh: trimesh.Trimesh,
    rh: trimesh.Trimesh,
    bs: trimesh.Trimesh | None,
    dest: Path,
    subj: str,
    space: str,
    surf_type: str,
    log_steps: list[str],
    log_files: list[str],
):
    lh_out = dest / f"{subj}_{space}_{surf_type}_LH.stl"
    rh_out = dest / f"{subj}_{space}_{surf_type}_RH.stl"
    lh.export(lh_out, file_type="stl")
    rh.export(rh_out, file_type="stl")
    log_steps += [f"Exported LH ⇒ {lh_out}", f"Exported RH ⇒ {rh_out}"]
    log_files += [str(lh_out), str(rh_out)]

    if bs:
        bs_out = dest / f"{subj}_{space}_brainstem.stl"
        bs.export(bs_out, file_type="stl")
        log_steps.append(f"Exported brainstem ⇒ {bs_out}")
        log_files.append(str(bs_out))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _build_parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # External‑tool sanity checks (only needed for MNI warps)
    # ---------------------------------------------------------------------- #
    if args.space.upper() == "MNI":
        require_cmds(
            ["antsApplyTransforms", "warpinit", "mrcat"],
            url_hint="Install ANTs & MRtrix3",
            logger=L,
        )

    # ---------------------------------------------------------------------- #
    # Structured run‑log dictionary
    # ---------------------------------------------------------------------- #
    runlog = {
        "tool": "brain_for_printing_cortical_surfaces",
        "subject_id": args.subject_id,
        "space": args.space,
        "surf_type": args.surf_type,
        "no_brainstem": args.no_brainstem,
        "split_hemis": args.split_hemis,
        "output_dir": str(out_dir),
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    # ---------------------------------------------------------------------- #
    # Work inside a managed temporary directory
    # ---------------------------------------------------------------------- #
    with temp_dir("cortical", keep=args.no_clean, base_dir=out_dir) as tmp_dir:
        L.info("Temporary folder: %s", tmp_dir)
        runlog["steps"].append(f"Created temp dir ⇒ {tmp_dir}")

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
            tmp_dir=tmp_dir,
        )
        runlog["steps"].append(f"Generated {args.surf_type} LH/RH in {args.space}")

        lh_mesh = meshes[f"{args.surf_type}_L"]
        rh_mesh = meshes[f"{args.surf_type}_R"]
        bs_mesh = meshes["brainstem"]  # may be None

        # ---------- export ----------
        if args.split_hemis:
            _export_split(
                lh_mesh,
                rh_mesh,
                bs_mesh,
                out_dir,
                args.subject_id,
                args.space,
                args.surf_type,
                runlog["steps"],
                runlog["output_files"],
            )
        else:
            combined: trimesh.Trimesh = lh_mesh + rh_mesh
            if bs_mesh:
                combined += bs_mesh
            out_path = (
                out_dir / f"{args.subject_id}_{args.space}_{args.surf_type}_brain.stl"
            )
            combined.export(out_path, file_type="stl")
            runlog["steps"].append(f"Exported merged mesh ⇒ {out_path}")
            runlog["output_files"].append(str(out_path))

        if args.no_clean:
            runlog["warnings"].append("Temp folder kept via --no_clean")
        else:
            runlog["steps"].append(f"Removed temp dir ⇒ {tmp_dir}")

    # ---------------------------------------------------------------------- #
    # Save JSON audit‑log
    # ---------------------------------------------------------------------- #
    write_log(runlog, out_dir, base_name="cortical_surfaces_log")
    L.info("Done.")


if __name__ == "__main__":
    main()

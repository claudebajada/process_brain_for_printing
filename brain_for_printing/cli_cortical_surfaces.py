# brain_for_printing/cli_cortical_surfaces.py

import os
import argparse
import uuid
import shutil
import trimesh

from .surfaces import generate_brain_surfaces
from .log_utils import write_log

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate cortical surfaces (LH + RH, optional brainstem) for a subject "
            "in either T1 or MNI space, optionally splitting hemispheres. "
            "No coloring is applied here; use brain_for_printing_color or "
            "brain_for_printing_overlay for coloring."
        )
    )

    parser.add_argument("--subjects_dir", required=True,
        help="Path to derivatives or subject data.")
    parser.add_argument("--subject_id", required=True,
        help="Subject identifier matching your derivatives naming.")
    parser.add_argument("--space", choices=["T1", "MNI"], default="T1",
        help="Output space: T1 (native) or MNI.")
    parser.add_argument("--surf_type", choices=["pial","white","mid"], default="pial",
        help="Which cortical surface to generate: pial, white, or mid (mid-thickness).")
    parser.add_argument("--output_dir", default=".",
        help="Where to store the output STL files (LH + RH + optional brainstem).")
    parser.add_argument("--no_brainstem", action="store_true",
        help="If set, skip extracting the brainstem.")
    parser.add_argument("--no_fill", action="store_true",
        help="If set, skip hole-filling in the extracted brainstem mesh.")
    parser.add_argument("--no_smooth", action="store_true",
        help="If set, skip Taubin smoothing in the extracted brainstem mesh.")
    parser.add_argument("--split_hemis", action="store_true",
        help="If set, export LH / RH / (optional) brainstem surfaces as separate files instead of merging.")
    parser.add_argument("--out_warp", default="warp.nii",
        help="(MNI only) Name of the 4D warp field to create, if needed.")
    parser.add_argument("--no_clean", action="store_true",
        help="If set, do NOT remove the temporary folder at the end.")
    parser.add_argument("--verbose", action="store_true",
        help="If set, prints additional progress messages.")
    parser.add_argument("--run", default=None,
        help="Run identifier, e.g. 'run-01' if your filenames include it.")
    parser.add_argument("--session", default=None,
        help="Session identifier, e.g. 'ses-01' if your filenames include it.")

    args = parser.parse_args()

    do_clean = not args.no_clean
    do_fill = not args.no_fill
    do_smooth = not args.no_smooth

    # Prepare a run log
    log = {
        "tool": "brain_for_printing_cortical_surfaces",
        "subject_id": args.subject_id,
        "space": args.space,
        "surf_type": args.surf_type,
        "no_brainstem": args.no_brainstem,
        "split_hemis": args.split_hemis,
        "output_dir": args.output_dir,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    # Create temporary folder for intermediate files
    tmp_dir = os.path.join(args.output_dir, f"_tmp_cortical_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    if args.verbose:
        print(f"[INFO] Temporary folder => {tmp_dir}")
    log["steps"].append(f"Created temp dir => {tmp_dir}")

    # Call the new utility to generate LH/RH (plus optional brainstem).
    # The user chooses pial, white, or mid via --surf_type.
    surfaces_tuple = (args.surf_type,)
    all_meshes = generate_brain_surfaces(
        subjects_dir=args.subjects_dir,
        subject_id=args.subject_id,
        space=args.space,
        surfaces=surfaces_tuple,
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
    bs_mesh = all_meshes["brainstem"]  # may be None if no_brainstem=True

    # Merge or split hemispheres
    if not args.split_hemis:
        # Combine LH + RH + optional brainstem
        final_mesh = lh_mesh + rh_mesh
        if bs_mesh:
            final_mesh += bs_mesh

        out_stl = os.path.join(args.output_dir, f"{args.subject_id}_{args.space}_{args.surf_type}_brain.stl")
        final_mesh.export(out_stl, file_type="stl")
        log["steps"].append("Exported combined LH+RH (and possibly brainstem)")
        log["output_files"].append(out_stl)
    else:
        # Export each hemisphere (and brainstem) separately
        lh_out = os.path.join(args.output_dir, f"{args.subject_id}_{args.space}_{args.surf_type}_LH.stl")
        rh_out = os.path.join(args.output_dir, f"{args.subject_id}_{args.space}_{args.surf_type}_RH.stl")
        lh_mesh.export(lh_out, file_type="stl")
        rh_mesh.export(rh_out, file_type="stl")
        log["steps"].append(f"Exported LH => {lh_out}")
        log["steps"].append(f"Exported RH => {rh_out}")
        log["output_files"].extend([lh_out, rh_out])

        if bs_mesh:
            bs_out = os.path.join(args.output_dir, f"{args.subject_id}_{args.space}_brainstem.stl")
            bs_mesh.export(bs_out, file_type="stl")
            log["steps"].append(f"Exported brainstem => {bs_out}")
            log["output_files"].append(bs_out)

    # Write the JSON log
    write_log(log, args.output_dir, base_name="cortical_surfaces_log")

    # Cleanup the temp folder if requested
    if do_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if args.verbose:
            print(f"[INFO] Removed temporary folder => {tmp_dir}")
        log["steps"].append(f"Removed temp dir => {tmp_dir}")
    else:
        if args.verbose:
            print(f"[INFO] Temporary folder retained => {tmp_dir}")


if __name__ == "__main__":
    main()

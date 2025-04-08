# brain_for_printing/cli_cortical_surfaces.py

import os
import argparse
import uuid
import shutil
import trimesh

from .io_utils import first_match
from .io_utils import flexible_match
from .mesh_utils import gifti_to_trimesh
from .warp_utils import generate_mrtrix_style_warp, warp_gifti_vertices
from .surfaces import extract_brainstem_in_t1, extract_brainstem_in_mni
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
    parser.add_argument("--output_dir", default=".",
        help="Where to store the output files (STL, etc.).")
    parser.add_argument("--use_white", action="store_true",
        help="Use white-matter surfaces instead of pial surfaces.")
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
        help="Run identifier, e.g., run-01 (optional)")
    parser.add_argument("--session", default=None, 
        help="Session identifier, e.g., ses-01 (optional)")
     
    args = parser.parse_args()

    do_fill = not args.no_fill
    do_smooth = not args.no_smooth
    do_clean = not args.no_clean

    # Create temporary folder
    tmp_dir = os.path.join(args.output_dir, f"_tmp_cortical_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)
    if args.verbose:
        print(f"[INFO] Temporary folder => {tmp_dir}")

    # Prepare a run log
    log = {
        "tool": "brain_for_printing_cortical_surfaces",
        "subject_id": args.subject_id,
        "space": args.space,
        "use_white": args.use_white,
        "no_brainstem": args.no_brainstem,
        "split_hemis": args.split_hemis,
        "output_dir": args.output_dir,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    # Determine the surface type to load
    surf_type = "smoothwm" if args.use_white else "pial"
    anat_dir = os.path.join(args.subjects_dir, args.subject_id, "anat")

    # Identify LH and RH surfaces (in T1 space)
    lh_surf_file = flexible_match(
        base_dir=anat_dir,
        subject_id=args.subject_id,
        descriptor=None,
        suffix=f"{surf_type}.surf",
        hemi="hemi-L",
        ext=".gii",
        run=args.run,  
        session=args.session 
    )

    rh_surf_file = flexible_match(
        base_dir=anat_dir,
        subject_id=args.subject_id,
        descriptor=None,
        suffix=f"{surf_type}.surf",
        hemi="hemi-R",
        ext=".gii",
        run=args.run,
        session=args.session
    )

    log["lh_surf_file"] = lh_surf_file
    log["rh_surf_file"] = rh_surf_file

    if args.space.upper() == "T1":
        # ==========================
        # T1-Space Surfaces
        # ==========================
        if args.verbose:
            print("[INFO] Loading LH T1 surface =>", lh_surf_file)
            print("[INFO] Loading RH T1 surface =>", rh_surf_file)

        lh_mesh = gifti_to_trimesh(lh_surf_file)
        rh_mesh = gifti_to_trimesh(rh_surf_file)
        log["steps"].append("Loaded LH/RH surfaces in T1 space")

        # Extract brainstem if requested
        bs_mesh = None
        if not args.no_brainstem:
            bs_gii = extract_brainstem_in_t1(
                subjects_dir=args.subjects_dir,
                subject_id=args.subject_id,
                tmp_dir=tmp_dir,
                verbose=args.verbose
            )
            log["steps"].append("Extracted brainstem in T1 space")

            bs_mesh = gifti_to_trimesh(bs_gii)
            if do_fill:
                trimesh.repair.fill_holes(bs_mesh)
                log["steps"].append("Filled holes in T1 brainstem mesh")
            if do_smooth:
                trimesh.smoothing.filter_taubin(bs_mesh, lamb=0.5, nu=-0.53, iterations=10)
                log["steps"].append("Smoothed T1 brainstem mesh (Taubin)")
            bs_mesh.invert()

        if not args.split_hemis:
            # Combine LH + RH + optional BS into one mesh
            final_mesh = lh_mesh + rh_mesh
            if bs_mesh:
                final_mesh += bs_mesh

            out_stl = os.path.join(args.output_dir, f"{args.subject_id}_T1_brain.stl")
            final_mesh.export(out_stl, file_type="stl")
            log["steps"].append("Exported combined T1 mesh (LH+RH+BS)")
            log["output_files"].append(out_stl)

        else:
            # Export LH, RH, and BS separately
            lh_out = os.path.join(args.output_dir, f"{args.subject_id}_T1_LH.stl")
            lh_mesh.export(lh_out, file_type="stl")
            log["steps"].append(f"Exported LH T1 surface => {lh_out}")
            log["output_files"].append(lh_out)

            rh_out = os.path.join(args.output_dir, f"{args.subject_id}_T1_RH.stl")
            rh_mesh.export(rh_out, file_type="stl")
            log["steps"].append(f"Exported RH T1 surface => {rh_out}")
            log["output_files"].append(rh_out)

            if bs_mesh:
                bs_out = os.path.join(args.output_dir, f"{args.subject_id}_T1_brainstem.stl")
                bs_mesh.export(bs_out, file_type="stl")
                log["steps"].append(f"Exported T1 brainstem => {bs_out}")
                log["output_files"].append(bs_out)

    else:
        # ==========================
        # MNI-Space Surfaces
        # ==========================
        # Generate the warp field to transform T1->MNI
        mni_file_pattern = f"{anat_dir}/*_run-01_space-MNI152NLin2009cAsym_*_T1w.nii.gz" # NEED TO FIX THESE TO BE FLEXIBLE
        t1_file_pattern  = f"{anat_dir}/*_run-01_desc-preproc_T1w.nii.gz"
        xfm_pattern      = f"{anat_dir}/*_run-01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"

        mni_file = first_match(mni_file_pattern)
        t1_file  = first_match(t1_file_pattern)
        xfm_file = first_match(xfm_pattern)

        warp_field = os.path.join(tmp_dir, args.out_warp)

        generate_mrtrix_style_warp(
            mni_file=mni_file,
            t1_file=t1_file,
            xfm_file=xfm_file,
            out_warp=args.out_warp,
            tmp_dir=tmp_dir,
            verbose=args.verbose
        )
        log["steps"].append("Generated MRtrix-style warp field for MNI")

        # Warp LH + RH to MNI
        lh_mni_out = os.path.join(tmp_dir, f"{args.subject_id}_L_{surf_type}_MNI.surf.gii")
        rh_mni_out = os.path.join(tmp_dir, f"{args.subject_id}_R_{surf_type}_MNI.surf.gii")

        warp_gifti_vertices(lh_surf_file, warp_field, lh_mni_out, verbose=args.verbose)
        warp_gifti_vertices(rh_surf_file, warp_field, rh_mni_out, verbose=args.verbose)
        log["steps"].append("Warped LH & RH surfaces into MNI space")

        lh_mesh = gifti_to_trimesh(lh_mni_out)
        rh_mesh = gifti_to_trimesh(rh_mni_out)

        # Extract brainstem in MNI if requested
        bs_mesh = None
        if not args.no_brainstem:
            out_aseg_in_mni = os.path.join(tmp_dir, "aseg_in_mni.nii.gz")
            bs_gii = extract_brainstem_in_mni(
                subjects_dir=args.subjects_dir,
                subject_id=args.subject_id,
                out_aseg_in_mni=out_aseg_in_mni,
                tmp_dir=tmp_dir,
                verbose=args.verbose
            )
            log["steps"].append("Extracted brainstem in MNI space")

            bs_mesh = gifti_to_trimesh(bs_gii)
            if do_fill:
                trimesh.repair.fill_holes(bs_mesh)
                log["steps"].append("Filled holes in MNI brainstem mesh")
            if do_smooth:
                trimesh.smoothing.filter_taubin(bs_mesh, lamb=0.5, nu=-0.53, iterations=10)
                log["steps"].append("Smoothed MNI brainstem mesh (Taubin)")
            bs_mesh.invert()

        if not args.split_hemis:
            # Combine LH + RH + optional BS
            final_mesh = lh_mesh + rh_mesh
            if bs_mesh:
                final_mesh += bs_mesh

            out_stl = os.path.join(args.output_dir, f"{args.subject_id}_MNI_brain.stl")
            final_mesh.export(out_stl, file_type="stl")
            log["steps"].append("Exported combined MNI mesh (LH+RH+BS)")
            log["output_files"].append(out_stl)

        else:
            # Export LH, RH, and BS separately
            lh_out = os.path.join(args.output_dir, f"{args.subject_id}_MNI_LH.stl")
            lh_mesh.export(lh_out, file_type="stl")
            log["steps"].append(f"Exported LH MNI surface => {lh_out}")
            log["output_files"].append(lh_out)

            rh_out = os.path.join(args.output_dir, f"{args.subject_id}_MNI_RH.stl")
            rh_mesh.export(rh_out, file_type="stl")
            log["steps"].append(f"Exported RH MNI surface => {rh_out}")
            log["output_files"].append(rh_out)

            if bs_mesh:
                bs_out = os.path.join(args.output_dir, f"{args.subject_id}_MNI_brainstem.stl")
                bs_mesh.export(bs_out, file_type="stl")
                log["steps"].append(f"Exported MNI brainstem => {bs_out}")
                log["output_files"].append(bs_out)

    # Write the JSON log
    out_dir = args.output_dir or "."
    write_log(log, output_dir=out_dir, base_name="cortical_surfaces_log")

    # Cleanup temp folder
    if not do_clean:
        if args.verbose:
            print(f"[INFO] Temporary folder retained => {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if args.verbose:
            print(f"[INFO] Removed temporary folder => {tmp_dir}")


if __name__ == "__main__":
    main()


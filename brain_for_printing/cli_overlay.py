# brain_for_printing/cli_overlay.py

import os
import argparse
import uuid
import trimesh

from .io_utils import load_nifti, first_match
from .color_utils import color_mesh_with_seg_and_param
from .log_utils import write_log

def main():
    parser = argparse.ArgumentParser(
        description="Color a mesh using segmentation and optionally overlay a parametric map."
    )
    parser.add_argument("--in_mesh", required=True, help="Input mesh (STL, OBJ, etc.)")
    parser.add_argument("--output", default="colored_overlay.obj", help="Output OBJ file")
    parser.add_argument("--segmentation", default=None, help="Optional segmentation NIfTI (5TT or aseg)")
    parser.add_argument("--param_map", default=None, help="Optional parametric map (NIfTI)")
    parser.add_argument("--param_threshold", type=float, default=0.0, help="Minimum value for param override")
    parser.add_argument("--num_colors", type=int, default=6, help="Number of discrete parametric colour bins")
    parser.add_argument("--subjects_dir", default=".", help="Used for fallback aseg search")
    parser.add_argument("--subject_id", default=None, help="Used for fallback aseg search")
    args = parser.parse_args()

    # Initialise log
    log = {
        "tool": "brain_for_printing_overlay",
        "in_mesh": args.in_mesh,
        "output": args.output,
        "segmentation": args.segmentation,
        "param_map": args.param_map,
        "param_threshold": args.param_threshold,
        "num_colors": args.num_colors,
        "steps": [],
        "warnings": [],
        "output_files": []
    }

    # Load mesh
    mesh = trimesh.load(args.in_mesh, force='mesh')
    if mesh.is_empty:
        raise ValueError("Mesh is empty or invalid")
    log["vertices"] = len(mesh.vertices)
    log["steps"].append("Loaded input mesh")

    # Load segmentation or fallback to fMRIPrep aseg
    if args.segmentation:
        seg_img, seg_affine = load_nifti(args.segmentation)
        log["steps"].append(f"Loaded segmentation from {args.segmentation}")
        log["segmentation_type"] = "5TT" if seg_img.ndim == 4 else "aseg (3D)"
    else:
        if not args.subject_id:
            raise ValueError("If no segmentation is given, --subject_id must be specified.")
        anat_dir = os.path.join(args.subjects_dir, args.subject_id, "anat")
        aseg_path = first_match(f"{anat_dir}/*_desc-aseg_dseg.nii.gz")
        seg_img, seg_affine = load_nifti(aseg_path)
        log["steps"].append(f"Loaded fallback aseg from {aseg_path}")
        log["segmentation_type"] = "aseg (3D)"

    # Load param map (optional)
    param_img = param_affine = None
    if args.param_map:
        param_img, param_affine = load_nifti(args.param_map)
        log["steps"].append(f"Loaded param map from {args.param_map}")

    # Apply colour logic
    colored = color_mesh_with_seg_and_param(
        mesh,
        segmentation_img=seg_img,
        seg_affine=seg_affine,
        param_img=param_img,
        param_affine=param_affine,
        threshold=args.param_threshold,
        num_colors=args.num_colors
    )
    log["steps"].append("Coloured mesh with segmentation and param map (if applicable)")

    # Export mesh
    colored.export(args.output, file_type='obj')
    log["steps"].append("Exported coloured mesh")
    log["output_files"].append(args.output)

    # Write log
    out_dir = os.path.dirname(args.output) or "."
    write_log(log, output_dir=out_dir, base_name="overlay_log")

if __name__ == "__main__":
    main()


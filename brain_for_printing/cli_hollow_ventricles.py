# brain_for_printing/cli_hollow_ventricles.py

import os
import argparse
import uuid
import shutil
import trimesh

from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .io_utils import first_match, run_cmd
from .log_utils import write_log


def is_engine_available(engine):
    if engine == "blender":
        return shutil.which("blender") is not None
    elif engine == "scad":
        return shutil.which("openscad") is not None
    return True


def repair_and_merge_components(mesh, log, voxel_pitch=1.0):
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        msg = f"Ventricle mesh has {len(components)} disconnected parts."
        print(f"[WARNING] {msg}")
        log["warnings"].append(msg)

    repaired = []
    for i, comp in enumerate(components):
        comp = comp.copy()

        # Basic cleaning
        comp.remove_degenerate_faces()
        comp.remove_unreferenced_vertices()
        comp.fix_normals()
        trimesh.repair.fill_holes(comp)
        trimesh.smoothing.filter_taubin(comp, lamb=0.5, nu=-0.53, iterations=10)

        if not comp.is_volume:
            print(f"[WARNING] Component {i} is not a volume. Voxelizing...")
            try:
                comp_voxel = comp.voxelized(pitch=voxel_pitch)
                comp = comp_voxel.as_boxes()
                log["steps"].append(f"Component {i} voxelized with pitch {voxel_pitch}")
            except Exception as e:
                msg = f"Voxelization failed on component {i}: {e}"
                print(f"[ERROR] {msg}")
                log["warnings"].append(msg)
                continue

        repaired.append(comp)

    merged = trimesh.util.concatenate(repaired)
    log["steps"].append("Repaired and merged ventricle components")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Subtract ventricles from a brain mesh using fMRIPrep aseg output."
    )
    parser.add_argument("--subjects_dir", required=True)
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--in_mesh", required=True)
    parser.add_argument("--space", choices=["T1", "MNI"], default="T1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", choices=["scad", "blender", "auto"], default="auto")
    parser.add_argument("--no_clean", action="store_true")
    args = parser.parse_args()

    tmp_dir = os.path.join(os.path.dirname(args.output), f"_tmp_hollow_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp_dir, exist_ok=True)

    log = {
        "tool": "brain_for_printing_hollow_ventricles",
        "subject_id": args.subject_id,
        "space": args.space,
        "in_mesh": args.in_mesh,
        "output": args.output,
        "engine": args.engine,
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    print(f"[INFO] Boolean engine selected: {args.engine}")
    print(f"[INFO] Blender available: {is_engine_available('blender')}")
    print(f"[INFO] OpenSCAD available: {is_engine_available('scad')}")

    # 1. Load brain mesh
    print(f"[INFO] Loading brain mesh => {args.in_mesh}")
    brain_mesh = trimesh.load(args.in_mesh, force='mesh')
    log["brain_vertices"] = len(brain_mesh.vertices)
    log["steps"].append("Loaded brain mesh")

    # 2. Find aseg
    anat_dir = os.path.join(args.subjects_dir, args.subject_id, "anat")
    aseg_pattern = f"{anat_dir}/*_run-01_desc-aseg_dseg.nii.gz"
    aseg_file = first_match(aseg_pattern)

    vent_mask_nii = os.path.join(tmp_dir, f"vent_mask_{args.space}.nii.gz")
    vent_labels = ["4", "5", "14", "15", "43", "44", "72"]

    if args.space == "T1":
        run_cmd([
            "mri_binarize",
            "--i", aseg_file,
            "--match", *vent_labels,
            "--o", vent_mask_nii
        ])
        log["steps"].append("Created ventricle binary mask using mri_binarize")
    else:
        raise NotImplementedError("MNI-space hollowing not implemented yet.")

    # 3. Convert to mesh
    vent_gii = os.path.join(tmp_dir, "ventricles.surf.gii")
    volume_to_gifti(vent_mask_nii, vent_gii, level=0.5)
    vent_mesh = gifti_to_trimesh(vent_gii)
    log["ventricles_vertices"] = len(vent_mesh.vertices)
    log["steps"].append("Converted ventricle mask to GIFTI surface")

    # 4. Split, repair, remesh with voxelization if needed
    vent_mesh = repair_and_merge_components(vent_mesh, log, voxel_pitch=1.0)

    # 5. Validate brain mesh
    if not brain_mesh.is_volume:
        msg = "Brain mesh is not a volume — boolean may fail."
        print(f"[WARNING] {msg}")
        log["warnings"].append(msg)

    # 6. Boolean subtraction
    print("[INFO] Performing boolean subtraction (brain - ventricles)...")
    trimesh.constants.DEFAULT_WEAK_ENGINE = args.engine

    try:
        hollowed = trimesh.boolean.difference(
            meshes=[brain_mesh, vent_mesh]
        )
        if hollowed.is_empty:
            raise ValueError("Boolean subtraction resulted in empty mesh.")
        log["steps"].append("Boolean subtraction successful")
    except Exception as e:
        print(f"[ERROR] Boolean subtraction failed: {e}")
        log["warnings"].append(f"Boolean subtraction failed: {str(e)}")
        write_log(log, os.path.dirname(args.output), base_name="hollow_log")
        raise

    # 7. Save result
    hollowed.export(args.output, file_type="stl")
    print(f"[INFO] Exported hollowed mesh => {args.output}")
    log["output_written"] = args.output
    log["result_vertices"] = len(hollowed.vertices)
    log["output_files"].append(args.output)

    # 8. Write log
    write_log(log, os.path.dirname(args.output), base_name="hollow_log")

    # 9. Cleanup
    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temp folder retained => {tmp_dir}")


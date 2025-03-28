# brain_for_printing/cli_hollow_ventricles.py

import os
import argparse
import uuid
import shutil
import trimesh
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from .mesh_utils import volume_to_gifti, gifti_to_trimesh
from .io_utils import first_match, run_cmd
from .log_utils import write_log

def is_engine_available(engine):
    if engine == "blender":
        return shutil.which("blender") is not None
    elif engine == "scad":
        return shutil.which("openscad") is not None
    return True

def repair_and_merge_components(mesh, log, voxel_pitch=1.0, min_volume_threshold=100.0):
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        msg = f"Ventricle mesh has {len(components)} disconnected parts."
        print(f"[WARNING] {msg}")
        log["warnings"].append(msg)

    repaired = []
    for i, comp in enumerate(components):
        comp = comp.copy()

        comp.remove_degenerate_faces()
        comp.remove_unreferenced_vertices()
        comp.fix_normals()
        trimesh.repair.fill_holes(comp)
        trimesh.smoothing.filter_taubin(comp, lamb=0.5, nu=-0.53, iterations=10)

        if not comp.is_volume:
            print(f"[WARNING] Component {i} is not a volume. Voxelizing...")
            try:
                comp_voxel = comp.voxelized(pitch=voxel_pitch)
                comp = comp_voxel.marching_cubes
                log["steps"].append(f"Component {i} voxelized with pitch {voxel_pitch}")
            except Exception as e:
                msg = f"Voxelization failed on component {i}: {e}"
                print(f"[ERROR] {msg}")
                log["warnings"].append(msg)
                continue

        if comp.volume < min_volume_threshold:
            print(f"[INFO] Skipping small component {i} (volume={comp.volume:.2f})")
            continue

        repaired.append(comp)

    if not repaired:
        raise ValueError("All ventricle components were discarded or failed to process.")

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
    parser.add_argument("--dilate_mask", action="store_true",
        help="Apply 3D dilation to the ventricle mask to improve surface integrity.")
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
    print(f"[INFO] Blender available: {shutil.which('blender') is not None}")
    print(f"[INFO] OpenSCAD available: {shutil.which('openscad') is not None}")

    print(f"[INFO] Loading brain mesh => {args.in_mesh}")
    brain_mesh = trimesh.load(args.in_mesh, force='mesh')
    
    print("[INFO] Mesh diagnostics:")
    print(f"  • Is watertight?   {brain_mesh.is_watertight}")
    print(f"  • Is a volume?     {brain_mesh.is_volume}")
    print(f"  • # Vertices:      {len(brain_mesh.vertices)}")
    print(f"  • # Faces:         {len(brain_mesh.faces)}")
    print(f"  • Surface area:    {brain_mesh.area:.2f} mm²")
    try:
        print(f"  • Enclosed volume: {brain_mesh.volume:.2f} mm³")
    except Exception as e:
        print(f"  • Enclosed volume: [error computing volume] {e}")

    
    log["brain_vertices"] = len(brain_mesh.vertices)
    log["steps"].append("Loaded brain mesh")

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

    if args.dilate_mask:
        print("[INFO] Applying dilation to ventricle mask...")
        nii = nib.load(vent_mask_nii)
        data = nii.get_fdata() > 0
        structure = generate_binary_structure(3, 2)
        dilated = binary_dilation(data, structure=structure, iterations=2)
        filled = binary_erosion(dilated, structure=structure, iterations=1)
        filled_img = nib.Nifti1Image(filled.astype(np.uint8), affine=nii.affine)
        vent_mask_nii = os.path.join(tmp_dir, "vent_mask_filled.nii.gz")
        nib.save(filled_img, vent_mask_nii)
        log["steps"].append("Applied dilation to ventricle mask")

    # Convert mask to surface
    vent_gii = os.path.join(tmp_dir, "ventricles.surf.gii")
    volume_to_gifti(vent_mask_nii, vent_gii, level=0.5)
    vent_mesh = gifti_to_trimesh(vent_gii)

    components = vent_mesh.split(only_watertight=False)

    filtered = []

    print(f"[INFO] Found {len(components)} ventricle components. Inspecting each:")

    for i, comp in enumerate(components):
        comp = comp.copy()
        comp.remove_degenerate_faces()
        comp.remove_unreferenced_vertices()
        comp.fix_normals()

        if comp.volume < 0:
            print(f"  • Component {i} has negative volume ({comp.volume:.2f}). Inverting...")
            comp.invert()    

        print(f"  • Component {i}: volume = {comp.volume:.2f} mm³ | watertight = {comp.is_watertight} | is_volume = {comp.is_volume}")

        if comp.volume < 1.0:
            print(f"    → Skipping: trivial volume")    
            continue

        if not comp.is_volume:
            print(f"    → Skipping: not a volume")
            continue

        print(f"    → Keeping ✅")
        filtered.append(comp)


    if not filtered:
        raise ValueError("No usable ventricle components found.")

    if not brain_mesh.is_volume:
        msg = "Brain mesh is not a volume — boolean may fail."
        print(f"[WARNING] {msg}")
        log["warnings"].append(msg)

    print("[INFO] Performing boolean subtraction (brain - ventricles)...")
    trimesh.constants.DEFAULT_WEAK_ENGINE = args.engine
    
    engine = args.engine
    if engine == "auto":
        engine = "trimesh"    
    
    hollowed = brain_mesh.copy()

    for i, vent in enumerate(filtered):
        print(f"[INFO] Subtracting component {i} (volume={vent.volume:.2f})...")
        try:
            #brain_path = os.path.join(tmp_dir, f"brain_before_component_{i}.stl")
            #vent_path = os.path.join(tmp_dir, f"vent_component_{i}.stl")
            #brain_mesh.export(brain_path)
            #vent.export(vent_path)

            # Make relative paths for OpenSCAD
            #rel_brain = f"brain_before_component_{i}.stl"
            #rel_vent = f"vent_component_{i}.stl"
            
            #scad_path = os.path.join(tmp_dir, f"subtract_{i}.scad")
            #with open(scad_path, "w") as f:
            #    f.write(f"""
            #difference() {{
            #  import("{rel_brain}");
            #  import("{rel_vent}");
            #}}""")
            
            #output_stl = os.path.join(tmp_dir, f"hollowed_component_{i}.stl")
            #print(f"[DEBUG] Running OpenSCAD: cd {tmp_dir} && openscad -o hollowed_component_{i}.stl subtract_{i}.scad")

            #os.system(f"cd {tmp_dir} && openscad -o hollowed_component_{i}.stl subtract_{i}.scad")
            
            
            hollowed = trimesh.boolean.difference([hollowed, vent], engine=engine)

            if not hollowed.is_volume or not hollowed.is_watertight:
                print(f"[WARNING] After subtracting component {i}, mesh is not watertight or not a volume.")
            else:
                print(f"[INFO] Subtraction of component {i} successful ✅")
        except Exception as e:
            print(f"[ERROR] Boolean subtraction failed for component {i}: {e}")
            continue


    print(f"[INFO] Hollowed mesh is watertight? {hollowed.is_watertight}")
    if not hollowed.is_watertight:
        print("[INFO] Attempting final voxel remeshing of hollowed mesh...")
        try:
            hollowed = hollowed.voxelized(pitch=0.5).fill().marching_cubes
            trimesh.repair.fill_holes(hollowed)
            hollowed.fix_normals()
            print(f"[INFO] Final mesh is watertight? {hollowed.is_watertight}")
        except Exception as e:
            print(f"[ERROR] Final voxel remeshing failed: {e}")
            raise

    if hollowed.is_empty:
        raise ValueError("Boolean subtraction resulted in empty mesh.")

    hollowed.export(args.output, file_type="stl")
    print(f"[INFO] Exported hollowed mesh => {args.output}")
    log["output_written"] = args.output
    log["result_vertices"] = len(hollowed.vertices)
    log["output_files"].append(args.output)

    write_log(log, os.path.dirname(args.output), base_name="hollow_log")

    if not args.no_clean:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"[INFO] Temp folder retained => {tmp_dir}")


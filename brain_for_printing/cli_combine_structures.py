#!/usr/bin/env python
# brain_for_printing/cli_combine_structures.py
#
# Combine FIRST‑/5ttgen‑derived subcortical & ventricular meshes and
# generate a smoothed cerebellar–WM surface.  Refactored for the 2025
# logging/ temp‑dir framework.

from __future__ import annotations
import argparse
import logging
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import vtk
from scipy.ndimage import binary_closing, label
import trimesh

from .io_utils import temp_dir, require_cmd
from .log_utils import get_logger, write_log
from .mesh_utils import volume_to_gifti, gifti_to_trimesh


# ─────────────────────────────────────────────────────────────────────────────
# vtk helpers (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
def _read_polydata(path: str) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def _flip_normals(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.FlipNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def _write_stl(poly: vtk.vtkPolyData, path: str, ascii: bool = True) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(path)
    writer.SetFileTypeToASCII() if ascii else writer.SetFileTypeToBinary()
    writer.Write()


def _append_polydata(polys: list[vtk.vtkPolyData]) -> vtk.vtkPolyData:
    append = vtk.vtkAppendPolyData()
    for p in polys:
        append.AddInputData(p)
    append.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(append.GetOutput())
    clean.Update()
    return clean.GetOutput()


def _is_valid(poly: vtk.vtkPolyData) -> bool:
    return poly.GetNumberOfPoints() > 0 and poly.GetNumberOfCells() > 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Combine subcortical + ventricular meshes and generate a separate "
            "cerebellar‑WM surface."
        )
    )
    ap.add_argument("--temp_dir", required=True, help="5ttgen temporary folder.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ascii", action="store_true", help="Export ASCII STL.")
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main() -> None:
    args = _parser().parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    L = get_logger(__name__, level=log_level)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runlog = {
        "tool": "brain_for_printing_combine_structures",
        "temp_dir": args.temp_dir,
        "output_dir": str(out_dir),
        "steps": [],
        "warnings": [],
        "output_files": [],
    }

    # optional external tool check: vtk is imported, but no binaries required.

    with temp_dir("combine", keep=args.no_clean, base_dir=out_dir) as tmp:
        tmp = Path(tmp)
        L.info("Temporary folder: %s", tmp)

        # ── Load FIRST subcortical meshes ────────────────────────────────────
        all_vtk = glob(str(Path(args.temp_dir) / "first-*_transformed.vtk"))
        subcorticals = sorted(f for f in all_vtk if not f.endswith("_first_transformed.vtk"))
        sub_meshes: list[vtk.vtkPolyData] = []
        skipped = 0

        for path in subcorticals:
            poly = _read_polydata(path)
            if _is_valid(poly):
                sub_meshes.append(poly)
            else:
                L.warning("Skipped empty subcortical: %s", path)
                skipped += 1
        runlog["steps"].append(f"Loaded {len(sub_meshes)} subcortical meshes")

        # ── Load ventricular & vessel meshes ────────────────────────────────
        vent_tags = ["Ventricle", "LatVent", "ChorPlex", "Inf-Lat-Vent", "vessel"]
        vent_cands = [
            f
            for f in glob(str(Path(args.temp_dir) / "*.vtk"))
            if any(tag in Path(f).name for tag in vent_tags)
            and not f.endswith("_init.vtk")
            and "_first" not in Path(f).name
            and "_transformed" not in Path(f).name
        ]
        vent_meshes: list[vtk.vtkPolyData] = []
        for path in vent_cands:
            poly = _flip_normals(_read_polydata(path))
            if _is_valid(poly):
                vent_meshes.append(poly)
            else:
                L.warning("Skipped empty vent/vessel: %s", path)
                skipped += 1
        runlog["steps"].append(f"Loaded {len(vent_meshes)} vent/vessel meshes")

        # ── Generate cerebellar‑WM surface ──────────────────────────────────
        cereb_pve = Path(args.temp_dir) / "T1_cerebellum_pve_2.nii.gz"
        nii = nib.load(cereb_pve)
        data = nii.get_fdata()
        aff = nii.affine

        binary = (data > 0.3).astype(np.uint8)
        binary = binary_closing(binary, structure=np.ones((3, 3, 3)))

        lbl, num = label(binary)
        counts = np.bincount(lbl.flat)
        counts[0] = 0
        largest = np.argmax(counts)
        cleaned = (lbl == largest).astype(np.uint8)

        cereb_bin = tmp / "cereb_wm_bin.nii.gz"
        nib.save(nib.Nifti1Image(cleaned, aff), cereb_bin)

        cereb_gii = tmp / "cereb_wm.surf.gii"
        volume_to_gifti(str(cereb_bin), str(cereb_gii), level=0.5)

        cereb_mesh = gifti_to_trimesh(str(cereb_gii))
        trimesh.smoothing.filter_taubin(cereb_mesh, lamb=0.5, nu=-0.53, iterations=10)
        cereb_mesh.remove_degenerate_faces()
        cereb_mesh.remove_unreferenced_vertices()
        cereb_mesh.fix_normals()

        cereb_stl = out_dir / "cerebellum_wm.stl"
        cereb_mesh.export(cereb_stl, file_type="stl")
        runlog["output_files"].append(str(cereb_stl))
        runlog["steps"].append("Generated cerebellar WM mesh")

        # ── Combine & export STLs ───────────────────────────────────────────
        combined_sub = _append_polydata(sub_meshes)
        combined_vent = _append_polydata(vent_meshes)
        combined_vent = _flip_normals(combined_vent)
        combined_all = _append_polydata(sub_meshes + [combined_vent])

        if _is_valid(combined_sub):
            out = out_dir / "combined_subcortical.stl"
            _write_stl(combined_sub, str(out), ascii=args.ascii)
            runlog["output_files"].append(str(out))
            runlog["steps"].append("Exported combined subcortical mesh")

        if _is_valid(combined_vent):
            out = out_dir / "combined_ventricles.stl"
            _write_stl(combined_vent, str(out), ascii=args.ascii)
            runlog["output_files"].append(str(out))
            runlog["steps"].append("Exported combined ventricle mesh")

        if _is_valid(combined_all):
            out = out_dir / "combined_all_structures.stl"
            _write_stl(combined_all, str(out), ascii=args.ascii)
            runlog["output_files"].append(str(out))
            runlog["steps"].append("Exported combined all‑structures mesh")

        runlog["skipped_count"] = skipped

    write_log(runlog, out_dir, base_name="combine_structures_log")
    L.info("Done.")


if __name__ == "__main__":
    main()

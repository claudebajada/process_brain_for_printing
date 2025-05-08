"""
5tt_utils.py
-----------
Utilities for working with 5ttgen (5-tissue-type) files.
Includes functions for running 5ttgen and loading subcortical meshes.
"""

import os
import logging
import subprocess
import json
import glob
from pathlib import Path
from typing import Dict, Optional, List, Union
import trimesh
import nibabel as nib
import numpy as np

# Handle Optional VTK Import
try:
    from vtk.util import numpy_support # type: ignore
    import vtk                     # type: ignore
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

L = logging.getLogger(__name__)

# VTK Helper Functions
if VTK_AVAILABLE:
    def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[vtk.vtkPolyData]:
        """
        Read VTK polydata from file using various reader types.
        
        Args:
            path: Path to VTK file
            logger: Logger instance
            
        Returns:
            Optional[vtk.vtkPolyData]: VTK polydata object if successful, None otherwise
        """
        logger.debug(f"Reading VTK: {path}")
        if not Path(path).exists():
            logger.error(f"VTK file not found: {path}")
            return None

        reader_types = [
            vtk.vtkSTLReader,
            vtk.vtkPolyDataReader,
            vtk.vtkXMLPolyDataReader,
            vtk.vtkGenericDataObjectReader
        ]

        for reader_class in reader_types:
            try:
                reader = reader_class()
                reader.SetFileName(path)
                reader.Update()

                poly_data_output = None
                if isinstance(reader, vtk.vtkGenericDataObjectReader):
                    if reader.IsFilePolyData():
                        poly_data_output = reader.GetPolyDataOutput()
                    else:
                        logger.debug(f"vtkGeneric reports {path} not PolyData.")
                        continue
                elif hasattr(reader, 'GetOutput') and isinstance(reader.GetOutput(), vtk.vtkPolyData):
                    poly_data_output = reader.GetOutput()

                if poly_data_output and poly_data_output.GetNumberOfPoints() > 0:
                    logger.info(f"Read {Path(path).name} via {reader_class.__name__}.")
                    return poly_data_output
                else:
                    logger.debug(f"{reader_class.__name__} gave no/empty PolyData for {path}.")
            except Exception as e_reader:
                logger.debug(f"{reader_class.__name__} failed for {path}: {e_reader}")

        logger.warning(f"Could not read valid PolyData from VTK file {path}.")
        return None

    def _vtk_polydata_to_trimesh(poly_data: Optional[vtk.vtkPolyData]) -> Optional[trimesh.Trimesh]:
        """
        Convert VTK polydata to trimesh object.
        
        Args:
            poly_data: VTK polydata object
            
        Returns:
            Optional[trimesh.Trimesh]: Trimesh object if successful, None otherwise
        """
        if not VTK_AVAILABLE or poly_data is None or poly_data.GetNumberOfPoints() == 0:
            return None

        num_pts = poly_data.GetPoints().GetNumberOfPoints()
        verts = np.zeros((num_pts, 3))
        for i in range(num_pts):
            verts[i, :] = poly_data.GetPoints().GetPoint(i)

        faces = []
        ids = vtk.vtkIdList()
        polys = poly_data.GetPolys()

        if polys and polys.GetNumberOfCells() > 0:
            polys.InitTraversal()
            while polys.GetNextCell(ids):
                faces.append([ids.GetId(j) for j in range(ids.GetNumberOfIds())])
        elif poly_data.GetStrips() and poly_data.GetStrips().GetNumberOfCells() > 0:
            strips = poly_data.GetStrips()
            strips.InitTraversal()
            L.warning("VTK has strips, may be lossy.")
            while strips.GetNextCell(ids):
                for j in range(ids.GetNumberOfIds() - 2):
                    faces.append([
                        ids.GetId(j + (j % 2)),
                        ids.GetId(j + 1 - (j % 2)),
                        ids.GetId(j + 2)
                    ])

        if not faces:
            L.warning("VTK has points but no faces.")
            return trimesh.Trimesh(vertices=verts)

        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
            return None if mesh.is_empty else mesh
        except Exception as e:
            L.error(f"Failed Trimesh creation: {e}")
            return None
else:
    def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[object]:
        """Stub for when VTK is not available."""
        logger.error(f"VTK unavailable, cannot read: {path}")
        return None

    def _vtk_polydata_to_trimesh(poly: Optional[object]) -> Optional[trimesh.Trimesh]:
        """Stub for when VTK is not available."""
        log_func = L.error if 'L' in globals() else print
        log_func("VTK unavailable.")
        return None

def run_5ttgen_hsvs_save_temp_bids(
    subject_id: str,
    fs_subject_dir: str,
    subject_work_dir: str,
    session_id: Optional[str] = None,
    nocrop: bool = True,
    sgm_amyg_hipp: bool = True,
    verbose: bool = False
) -> Optional[str]:
    """
    Run 5ttgen hsvs and save results in BIDS format.
    
    Args:
        subject_id: Subject ID (e.g., 'sub-01')
        fs_subject_dir: FreeSurfer subject directory
        subject_work_dir: Working directory for temporary files
        session_id: BIDS session ID
        nocrop: Whether to disable cropping
        sgm_amyg_hipp: Whether to include subcortical gray matter, amygdala, and hippocampus
        verbose: Enable verbose logging
        
    Returns:
        Optional[str]: Path to output directory if successful, None otherwise
    """
    if not VTK_AVAILABLE:
        L.error("VTK not available, cannot run 5ttgen")
        return None
        
    subject_id_clean = subject_id.replace('sub-', '')
    work_dir = Path(subject_work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct 5ttgen command
    cmd = ["5ttgen", "hsvs", fs_subject_dir]
    if nocrop:
        cmd.append("-nocrop")
    if sgm_amyg_hipp:
        cmd.append("-sgm_amyg_hipp")
    cmd.extend(["-tempdir", str(work_dir)])
    
    # Add session ID if provided
    if session_id:
        cmd.extend(["-session", session_id])
        
    # Run 5ttgen
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        L.error(f"5ttgen failed: {e}")
        if not verbose:
            L.error(f"stdout: {e.stdout.decode()}")
            L.error(f"stderr: {e.stderr.decode()}")
        return None
        
    # Save results in BIDS format
    output_dir = work_dir / f"sub-{subject_id_clean}" / "anat"
    if session_id:
        output_dir = output_dir / f"ses-{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and rename files
    for f in work_dir.glob("5tt_*.nii.gz"):
        new_name = f"sub-{subject_id_clean}"
        if session_id:
            new_name += f"_ses-{session_id}"
        new_name += "_desc-5tt_dseg.nii.gz"
        os.rename(f, output_dir / new_name)
        
    return str(output_dir)

def load_subcortical_and_ventricle_meshes(five_ttgen_persistent_dir_str: str) -> Dict[str, trimesh.Trimesh]:
    """
    Load subcortical and ventricle meshes from 5ttgen output.
    
    Args:
        five_ttgen_persistent_dir_str: Path to 5ttgen output directory
        
    Returns:
        Dict[str, trimesh.Trimesh]: Dictionary of mesh names to trimesh objects
    """
    if not VTK_AVAILABLE:
        L.error("VTK not available, cannot load meshes")
        return {}
        
    result: Dict[str, trimesh.Trimesh] = {}
    five_ttgen_persistent_dir = Path(five_ttgen_persistent_dir_str)
    
    # Find the 5ttgen-tmp directory
    tmp_dirs = list(five_ttgen_persistent_dir.glob("5ttgen-tmp-*"))
    if not tmp_dirs:
        L.error(f"No 5ttgen-tmp directory found in {five_ttgen_persistent_dir}")
        return {}
    
    tmp_dir = tmp_dirs[0]
    L.info(f"Found 5ttgen tmp directory: {tmp_dir}")
    
    # Load subcortical meshes (first-*_transformed.vtk)
    for mesh_path in tmp_dir.glob("first-*_transformed.vtk"):
        try:
            poly_data = _read_vtk_polydata(str(mesh_path), L)
            if poly_data is None:
                continue
                
            mesh = _vtk_polydata_to_trimesh(poly_data)
            if mesh is None or mesh.is_empty:
                continue
                
            # Get mesh name from filename
            # Convert first-L_Puta_transformed.vtk -> L_Puta
            mesh_name = mesh_path.stem.replace("first-", "").replace("_transformed", "")
            result[f"subcortical-{mesh_name}"] = mesh
            L.info(f"Loaded subcortical mesh: {mesh_name}")
            
        except Exception as e:
            L.error(f"Failed to load subcortical mesh {mesh_path}: {e}")
            continue
            
    # Load ventricle meshes (CSF.vtk, *-Ventricle.vtk, etc.)
    ventricle_patterns = [
        "CSF.vtk",
        "*-Ventricle.vtk",
        "*_LatVent_ChorPlex.vtk",
        "*-Inf-Lat-Vent.vtk"
    ]
    
    for pattern in ventricle_patterns:
        for mesh_path in tmp_dir.glob(pattern):
            try:
                poly_data = _read_vtk_polydata(str(mesh_path), L)
                if poly_data is None:
                    continue
                    
                mesh = _vtk_polydata_to_trimesh(poly_data)
                if mesh is None or mesh.is_empty:
                    continue
                    
                # Get mesh name from filename
                # Remove common suffixes and prefixes
                mesh_name = mesh_path.stem
                mesh_name = mesh_name.replace("_init", "")
                mesh_name = mesh_name.replace("_ChorPlex", "")
                mesh_name = mesh_name.replace("-Inf-Lat-Vent", "")
                mesh_name = mesh_name.replace("-Ventricle", "")
                
                # Add ventricle prefix
                result[f"ventricle-{mesh_name}"] = mesh
                L.info(f"Loaded ventricle mesh: {mesh_name}")
                
            except Exception as e:
                L.error(f"Failed to load ventricle mesh {mesh_path}: {e}")
                continue
            
    return result 
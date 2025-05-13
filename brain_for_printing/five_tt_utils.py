"""
5tt_utils.py
-----------
Utilities for working with 5ttgen (5-tissue-type) files.
Includes functions for running 5ttgen and loading subcortical meshes.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Union 
import trimesh
import numpy as np   

VTK_AVAILABLE = False
try:
    from vtk.util import numpy_support # type: ignore 
    import vtk                     # type: ignore
    VTK_AVAILABLE = True
except ImportError:
    pass 

L = logging.getLogger(__name__)

def is_vtk_available() -> bool:
    return VTK_AVAILABLE

def _try_vtk_reader(reader_instance, file_path_str: str, reader_name: str, logger: logging.Logger) -> Optional[object]:
    try:
        reader_instance.SetFileName(file_path_str)
        reader_instance.Update()
        poly_data_output = None
        if isinstance(reader_instance, vtk.vtkGenericDataObjectReader):
            if reader_instance.IsFilePolyData():
                poly_data_output = reader_instance.GetPolyDataOutput()
            else:
                logger.debug(f"{reader_name} determined {Path(file_path_str).name} is not PolyData.")
                return None
        elif hasattr(reader_instance, 'GetOutput') and isinstance(reader_instance.GetOutput(), vtk.vtkPolyData):
            poly_data_output = reader_instance.GetOutput()
        if poly_data_output and poly_data_output.GetNumberOfPoints() > 0:
            logger.info(f"Successfully read {Path(file_path_str).name} using {reader_name}.")
            return poly_data_output
        else:
            logger.debug(f"{reader_name} read file but resulted in no points or not PolyData for {Path(file_path_str).name}.")
            return None
    except Exception as e:
        logger.debug(f"{reader_name} failed for {Path(file_path_str).name}: {e}")
        return None

def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[object]:
    if not is_vtk_available():
        logger.error(f"VTK is not available, cannot read VTK file: {path}")
        return None
    file_path_obj = Path(path)
    if not file_path_obj.exists():
        logger.error(f"VTK file not found: {path}")
        return None
    logger.debug(f"Attempting to read VTK file: {file_path_obj.name}")
    file_ext = file_path_obj.suffix.lower()
    poly_data = None
    if file_ext == '.vtk':
        logger.debug(f"Detected .vtk extension. Trying vtkPolyDataReader first.")
        poly_data = _try_vtk_reader(vtk.vtkPolyDataReader(), path, "vtkPolyDataReader", logger)
        if not poly_data: 
            logger.debug(f"vtkPolyDataReader failed for {file_path_obj.name}, trying vtkGenericDataObjectReader as fallback.")
            poly_data = _try_vtk_reader(vtk.vtkGenericDataObjectReader(), path, "vtkGenericDataObjectReader (for .vtk)", logger)
    elif file_ext == '.stl': # ... (stl handling) ...
        logger.debug(f"Detected .stl extension. Trying vtkSTLReader first.")
        poly_data = _try_vtk_reader(vtk.vtkSTLReader(), path, "vtkSTLReader", logger)
        if not poly_data:
            logger.debug(f"vtkSTLReader failed for {file_path_obj.name}, trying vtkGenericDataObjectReader as fallback.")
            poly_data = _try_vtk_reader(vtk.vtkGenericDataObjectReader(), path, "vtkGenericDataObjectReader (for .stl)", logger)
    elif file_ext == '.vtp': # ... (vtp handling) ...
        logger.debug(f"Detected .vtp extension. Trying vtkXMLPolyDataReader first.")
        poly_data = _try_vtk_reader(vtk.vtkXMLPolyDataReader(), path, "vtkXMLPolyDataReader", logger)
        if not poly_data:
            logger.debug(f"vtkXMLPolyDataReader failed for {file_path_obj.name}, trying vtkGenericDataObjectReader as fallback.")
            poly_data = _try_vtk_reader(vtk.vtkGenericDataObjectReader(), path, "vtkGenericDataObjectReader (for .vtp)", logger)
    else: 
        logger.debug(f"Unknown or unhandled extension '{file_ext}'. Trying vtkGenericDataObjectReader first.")
        poly_data = _try_vtk_reader(vtk.vtkGenericDataObjectReader(), path, "vtkGenericDataObjectReader (unknown ext)", logger)
        if not poly_data:
            logger.debug(f"vtkGenericDataObjectReader failed for {file_path_obj.name}. Trying vtkPolyDataReader as broad fallback.")
            if poly_data is None or not isinstance(poly_data, vtk.vtkPolyData): 
                 poly_data = _try_vtk_reader(vtk.vtkPolyDataReader(), path, "vtkPolyDataReader (fallback)", logger)
    if not poly_data:
        logger.warning(f"Could not read valid PolyData from {file_path_obj.name} with prioritized or fallback readers.")
    return poly_data

def _vtk_polydata_to_trimesh(poly_data: Optional[object]) -> Optional[trimesh.Trimesh]:
    if not is_vtk_available(): 
        L.error("VTK is not available, cannot convert polydata to Trimesh.")
        return None
    if poly_data is None:
        L.debug("No polydata provided to _vtk_polydata_to_trimesh.")
        return None
    if not isinstance(poly_data, vtk.vtkPolyData):
        L.error(f"Input to _vtk_polydata_to_trimesh is not vtkPolyData, but {type(poly_data)}.")
        return None
    num_pts = poly_data.GetNumberOfPoints()
    if num_pts == 0:
        L.warning("Polydata has no points, resulting Trimesh will be empty.")
        return trimesh.Trimesh() 
    points_vtk = poly_data.GetPoints()
    verts_np = numpy_support.vtk_to_numpy(points_vtk.GetData())
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(poly_data)
    triangle_filter.Update()
    triangulated_poly_data = triangle_filter.GetOutput()
    polys_vtk = triangulated_poly_data.GetPolys()
    if polys_vtk.GetNumberOfCells() == 0:
        obj_name_info = "N/A"
        if poly_data.GetInformation() and poly_data.GetInformation().Has(vtk.vtkDataObject.DATA_OBJECT_FIELD()):
            obj_name_info = poly_data.GetInformation().Get(vtk.vtkDataObject.DATA_OBJECT_FIELD())
        L.warning(f"Polydata (name: {obj_name_info}) has points but no cells (faces) after triangulation. Creating point cloud.")
        return trimesh.Trimesh(vertices=verts_np) 
    faces_vtk_flat_array = numpy_support.vtk_to_numpy(polys_vtk.GetData())
    faces_np_list = [] 
    i = 0
    num_cells_processed = 0
    while i < len(faces_vtk_flat_array):
        num_pts_in_face = int(faces_vtk_flat_array[i]) 
        if num_pts_in_face != 3: 
            L.warning(f"Encountered a non-triangular polygon ({num_pts_in_face} points) in supposedly triangulated mesh. Skipping this face. Cell index {num_cells_processed}.")
            i += (1 + num_pts_in_face)
            num_cells_processed+=1
            continue
        if i + num_pts_in_face >= len(faces_vtk_flat_array):
            L.warning(f"Face data indicates {num_pts_in_face} points, but array is too short. Index i={i}, array length={len(faces_vtk_flat_array)}. Skipping face.")
            break 
        p0_idx, p1_idx, p2_idx = int(faces_vtk_flat_array[i+1]), int(faces_vtk_flat_array[i+2]), int(faces_vtk_flat_array[i+3])
        if not (0 <= p0_idx < num_pts and 0 <= p1_idx < num_pts and 0 <= p2_idx < num_pts):
            L.warning(f"Invalid vertex index found in face data: {(p0_idx, p1_idx, p2_idx)} for {num_pts} vertices. Skipping face.")
            i += (1 + num_pts_in_face)
            num_cells_processed+=1
            continue
        faces_np_list.append([p0_idx, p1_idx, p2_idx])
        i += (1 + num_pts_in_face)
        num_cells_processed+=1
    if not faces_np_list: 
        L.warning("No valid triangular faces found after processing polygons. Resulting Trimesh may be empty or a point cloud.")
        return trimesh.Trimesh(vertices=verts_np)
    try:
        mesh = trimesh.Trimesh(vertices=verts_np, faces=np.array(faces_np_list), process=True) 
        return None if mesh.is_empty else mesh
    except Exception as e: 
        L.error(f"Failed to create Trimesh object from VTK polydata: {e}", exc_info=True) 
        return None

def run_5ttgen_hsvs_save_temp_bids(
    subject_id: str,
    fs_subject_dir: str,       
    subject_work_dir: str,     
    session_id: Optional[str] = None,
    nocrop: bool = True,       
    sgm_amyg_hipp: bool = True,
    verbose: bool = False
) -> bool: 
    """
    Runs 5ttgen hsvs. The primary goal is to generate the '5ttgen-tmp-XXXXXX'
    directory (within 'subject_work_dir') which contains VTK surface meshes.
    The main 5TT NIfTI image is also generated but not explicitly managed by this function.
    Includes -force to overwrite existing 5ttgen NIfTI output if called.
    """
    subject_id_clean = subject_id.replace('sub-', '') 
    work_dir_path = Path(subject_work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)
    L.info(f"5ttgen will use base work directory: {work_dir_path} (expects '5ttgen-tmp-*' inside)")
    
    temp_5tt_output_name = f"5tt_output_for_{subject_id_clean}.nii.gz" 
    
    cmd = [
        "5ttgen", "hsvs", str(Path(fs_subject_dir)),
        str(work_dir_path / temp_5tt_output_name) 
    ]
    
    if nocrop:
        cmd.append("-nocrop")
    if sgm_amyg_hipp: 
        cmd.append("-sgm_amyg_hipp")
    
    cmd.extend(["-tempdir", str(work_dir_path)])
    cmd.append("-force") # *** ADDED -force FLAG ***
    
    L.info(f"Running 5ttgen command: {' '.join(cmd)}")
    try:
        from .io_utils import run_cmd as run_external_cmd 
        run_external_cmd(cmd, verbose=verbose) 
        L.info(f"5ttgen command completed. Temporary VTK files should be in a '5ttgen-tmp-*' subdirectory within {work_dir_path}.")
        if not list(work_dir_path.glob("5ttgen-tmp-*")):
            L.warning(f"5ttgen command ran but no '5ttgen-tmp-*' directory found in {work_dir_path}. VTK files might be missing.")
        return True
    except RuntimeError as e: 
        L.error(f"5ttgen command failed: {e}")
        return False
    except FileNotFoundError: 
        L.error("5ttgen command not found. Please ensure MRtrix3 is installed and in PATH.")
        return False
        
def load_subcortical_and_ventricle_meshes(five_ttgen_base_work_dir_str: str) -> Dict[str, trimesh.Trimesh]:
    if not is_vtk_available():
        L.error("VTK is not available, cannot load 5ttgen VTK meshes.")
        return {}
    result: Dict[str, trimesh.Trimesh] = {}
    five_ttgen_base_work_dir = Path(five_ttgen_base_work_dir_str)
    tmp_dirs_found = list(five_ttgen_base_work_dir.glob("5ttgen-tmp-*"))
    if not tmp_dirs_found:
        L.warning(f"No '5ttgen-tmp-*' directory found within {five_ttgen_base_work_dir}. Cannot load VTK meshes.")
        return {}
    actual_5ttgen_tmp_dir = tmp_dirs_found[0] 
    L.info(f"Found 5ttgen temporary data directory for VTKs: {actual_5ttgen_tmp_dir}")
    for vtk_file_path in actual_5ttgen_tmp_dir.glob("first-*_transformed.vtk"):
        mesh_name_base = vtk_file_path.stem.replace("first-", "").replace("_transformed", "")
        mesh_key = f"subcortical-{mesh_name_base}" 
        poly_data = _read_vtk_polydata(str(vtk_file_path), L)
        if poly_data:
            mesh = _vtk_polydata_to_trimesh(poly_data)
            if mesh and not mesh.is_empty:
                result[mesh_key] = mesh
                L.debug(f"Loaded subcortical mesh: {mesh_key} from {vtk_file_path.name}")
            else: L.warning(f"Failed to convert or got empty Trimesh for {mesh_key} from {vtk_file_path.name}")
        else: L.warning(f"Failed to read VTK polydata for {mesh_key} from {vtk_file_path.name}")
    ventricle_patterns = [ "CSF.vtk", "*-Ventricle.vtk", "*_LatVent_ChorPlex.vtk", "*-Inf-Lat-Vent.vtk" ]
    processed_ventricle_files = set()
    for pattern in ventricle_patterns:
        for vtk_file_path in actual_5ttgen_tmp_dir.glob(pattern):
            if vtk_file_path in processed_ventricle_files: continue 
            processed_ventricle_files.add(vtk_file_path)
            mesh_name_base = vtk_file_path.stem
            mesh_name_base = mesh_name_base.replace("_init", "").replace("_ChorPlex", "").replace("-Ventricle", "_Ventricle")
            if "LatVent" in mesh_name_base and "Inf" not in mesh_name_base: 
                 mesh_name_base = mesh_name_base.replace("LatVent","LateralVentricle")
            mesh_key = f"ventricle-{mesh_name_base}" 
            poly_data = _read_vtk_polydata(str(vtk_file_path), L)
            if poly_data:
                mesh = _vtk_polydata_to_trimesh(poly_data)
                if mesh and not mesh.is_empty:
                    result[mesh_key] = mesh
                    L.debug(f"Loaded ventricle/CSF mesh: {mesh_key} from {vtk_file_path.name}")
                else: L.warning(f"Failed to convert or got empty Trimesh for {mesh_key} from {vtk_file_path.name}")
            else: L.warning(f"Failed to read VTK polydata for {mesh_key} from {vtk_file_path.name}")
    if not result: L.warning(f"No subcortical or ventricle meshes were successfully loaded from {actual_5ttgen_tmp_dir}.")
    else: L.info(f"Successfully loaded {len(result)} subcortical/ventricle meshes: {list(result.keys())}")
    return result

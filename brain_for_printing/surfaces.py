# brain_for_printing/surfaces.py

"""
surfaces.py
-----------
This module serves as a compatibility layer, re-exporting functions from specialized modules:
- surfgen_utils.py: Core surface generation functionality
- aseg_utils.py: ASEG-specific functions
- five_tt_utils.py: 5ttgen-related functions

It also provides VTK-related functionality for mesh handling.
"""
import logging
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .surfgen_utils import generate_brain_surfaces
from .aseg_utils import extract_structure_surface
from .five_tt_utils import run_5ttgen_hsvs_save_temp_bids, load_subcortical_and_ventricle_meshes

L = logging.getLogger(__name__)

# Re-export functions from other modules for backward compatibility
__all__ = [
    'generate_brain_surfaces',
    'extract_structure_surface',
    'run_5ttgen_hsvs_save_temp_bids',
    'load_subcortical_and_ventricle_meshes'
]

# Handle Optional VTK Import
try:
    from vtk.util import numpy_support # type: ignore
    import vtk                     # type: ignore
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

# Define stubs if VTK not available *before* they are potentially used
if not VTK_AVAILABLE:
    def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[object]:
        logger.error(f"VTK unavailable, cannot read: {path}")
        return None

    def _vtk_polydata_to_trimesh(poly: Optional[object]) -> Optional[trimesh.Trimesh]:
        log_func = L.error if 'L' in globals() else print
        log_func("VTK unavailable.")
        return None

# VTK Helper Functions
if VTK_AVAILABLE:
    def _read_vtk_polydata(path: str, logger: logging.Logger) -> Optional[vtk.vtkPolyData]:
        """Read VTK polydata from file using various reader types."""
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
        """Convert VTK polydata to trimesh object."""
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

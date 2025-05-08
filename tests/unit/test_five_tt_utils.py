"""
Unit tests for five_tt_utils.py
"""
import pytest
from pathlib import Path
import logging
import trimesh
import numpy as np

from brain_for_printing.five_tt_utils import (
    _read_vtk_polydata,
    _vtk_polydata_to_trimesh,
    load_subcortical_and_ventricle_meshes
)

def test_read_vtk_polydata_with_vtk_available(sample_vtk_file, mock_vtk_available):
    """Test reading VTK file when VTK is available."""
    logger = logging.getLogger(__name__)
    poly_data = _read_vtk_polydata(sample_vtk_file, logger)
    assert poly_data is not None
    assert poly_data.GetNumberOfPoints() == 8  # Cube has 8 vertices
    assert poly_data.GetNumberOfPolys() == 6  # Cube has 6 faces

def test_read_vtk_polydata_with_vtk_unavailable(sample_vtk_file, mock_vtk_unavailable):
    """Test reading VTK file when VTK is unavailable."""
    logger = logging.getLogger(__name__)
    poly_data = _read_vtk_polydata(sample_vtk_file, logger)
    assert poly_data is None

def test_vtk_polydata_to_trimesh_with_vtk_available(sample_vtk_file, mock_vtk_available):
    """Test converting VTK polydata to trimesh when VTK is available."""
    logger = logging.getLogger(__name__)
    poly_data = _read_vtk_polydata(sample_vtk_file, logger)
    assert poly_data is not None
    
    mesh = _vtk_polydata_to_trimesh(poly_data)
    assert mesh is not None
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) == 8  # Cube has 8 vertices
    assert len(mesh.faces) == 6  # Cube has 6 faces

def test_vtk_polydata_to_trimesh_with_vtk_unavailable(mock_vtk_unavailable):
    """Test converting VTK polydata to trimesh when VTK is unavailable."""
    mesh = _vtk_polydata_to_trimesh(None)
    assert mesh is None

def test_load_subcortical_and_ventricle_meshes_with_vtk_unavailable(temp_dir, mock_vtk_unavailable):
    """Test loading meshes when VTK is unavailable."""
    meshes = load_subcortical_and_ventricle_meshes(temp_dir)
    assert meshes == {}

def test_load_subcortical_and_ventricle_meshes_empty_dir(temp_dir, mock_vtk_available):
    """Test loading meshes from an empty directory."""
    meshes = load_subcortical_and_ventricle_meshes(temp_dir)
    assert meshes == {} 
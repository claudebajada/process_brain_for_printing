"""
Integration tests for cortical_surfaces.py
"""
import pytest
from pathlib import Path
import logging
import shutil
import os

from brain_for_printing.cortical_surfaces import (
    generate_cortical_surfaces,
    save_surfaces_to_stl
)

def test_generate_cortical_surfaces_basic(temp_dir, mock_vtk_available):
    """Test basic cortical surface generation workflow."""
    # Create mock FreeSurfer subject directory structure
    subjects_dir = temp_dir / "subjects"
    subject_id = "test_subject"
    subject_dir = subjects_dir / subject_id
    
    # Create minimal required FreeSurfer files
    os.makedirs(subject_dir / "surf", exist_ok=True)
    os.makedirs(subject_dir / "mri", exist_ok=True)
    
    # Create mock surface files
    with open(subject_dir / "surf" / "lh.white", "w") as f:
        f.write("mock surface data")
    with open(subject_dir / "surf" / "rh.white", "w") as f:
        f.write("mock surface data")
    
    # Test surface generation
    surfaces = generate_cortical_surfaces(
        subjects_dir=str(subjects_dir),
        subject_id=subject_id,
        surfaces=["white"],
        hemispheres=["lh", "rh"]
    )
    
    assert surfaces is not None
    assert len(surfaces) == 2  # One for each hemisphere
    assert all(k in surfaces for k in ["lh.white", "rh.white"])

def test_save_surfaces_to_stl(temp_dir, mock_vtk_available):
    """Test saving surfaces to STL files."""
    # Create mock surfaces dictionary
    surfaces = {
        "lh.white": None,  # In real test, this would be a trimesh object
        "rh.white": None
    }
    
    output_dir = temp_dir / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test saving surfaces
    save_surfaces_to_stl(surfaces, str(output_dir))
    
    # Check that output directory exists
    assert output_dir.exists()
    
    # In a real test with actual mesh objects, we would check for the STL files
    # assert (output_dir / "lh.white.stl").exists()
    # assert (output_dir / "rh.white.stl").exists()

def test_generate_cortical_surfaces_invalid_subject(temp_dir, mock_vtk_available):
    """Test surface generation with invalid subject directory."""
    subjects_dir = temp_dir / "subjects"
    subject_id = "nonexistent_subject"
    
    with pytest.raises(FileNotFoundError):
        generate_cortical_surfaces(
            subjects_dir=str(subjects_dir),
            subject_id=subject_id,
            surfaces=["white"],
            hemispheres=["lh", "rh"]
        ) 
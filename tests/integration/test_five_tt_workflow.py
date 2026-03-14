"""
Integration tests for 5ttgen workflow
"""
import pytest
from pathlib import Path
import logging
import shutil
import os
import json

from brain_for_printing.five_tt_utils import (
    run_5ttgen_hsvs_save_temp_bids,
    load_subcortical_and_ventricle_meshes
)

def test_run_5ttgen_hsvs_save_temp_bids(temp_dir, mock_vtk_available):
    """Test running 5ttgen and saving results in BIDS format."""
    # Create mock FreeSurfer subject directory structure
    subjects_dir = temp_dir / "subjects"
    subject_id = "test_subject"
    subject_dir = subjects_dir / subject_id
    
    # Create minimal required FreeSurfer files
    os.makedirs(subject_dir / "mri", exist_ok=True)
    with open(subject_dir / "mri" / "T1.mgz", "w") as f:
        f.write("mock T1 data")
    
    # Test running 5ttgen
    output_dir = temp_dir / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # In a real test, we would mock the subprocess call to 5ttgen
    # For now, we'll just test the function signature and error handling
    with pytest.raises(FileNotFoundError):  # Since 5ttgen won't be available
        run_5ttgen_hsvs_save_temp_bids(
            subject_id=subject_id,
            subjects_dir=str(subjects_dir),
            output_dir=str(output_dir),
            crop=True,
            include_subcortical=True
        )

def test_load_subcortical_and_ventricle_meshes_with_mock_files(temp_dir, mock_vtk_available):
    """Test loading subcortical and ventricle meshes with mock VTK files."""
    # Create mock 5ttgen output directory structure
    work_dir = temp_dir / "work"
    os.makedirs(work_dir, exist_ok=True)
    
    # Create mock VTK files
    vtk_dir = work_dir / "5ttgen-tmp-TEST"
    os.makedirs(vtk_dir, exist_ok=True)
    
    # Create mock subcortical and ventricle files
    with open(vtk_dir / "first-L_Puta_transformed.vtk", "w") as f:
        f.write("mock VTK data")
    with open(vtk_dir / "CSF.vtk", "w") as f:
        f.write("mock VTK data")
    
    # Test loading meshes
    meshes = load_subcortical_and_ventricle_meshes(str(work_dir))
    
    # In a real test with actual VTK files, we would check the loaded meshes
    # For now, we'll just verify the function runs without errors
    assert isinstance(meshes, dict)

def test_run_5ttgen_invalid_subject(temp_dir, mock_vtk_available):
    """Test running 5ttgen with invalid subject directory."""
    subjects_dir = temp_dir / "subjects"
    subject_id = "nonexistent_subject"
    output_dir = temp_dir / "output"
    
    with pytest.raises(FileNotFoundError):
        run_5ttgen_hsvs_save_temp_bids(
            subject_id=subject_id,
            subjects_dir=str(subjects_dir),
            output_dir=str(output_dir),
            crop=True,
            include_subcortical=True
        ) 
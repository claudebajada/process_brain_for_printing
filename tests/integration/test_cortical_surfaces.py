"""
Integration tests for cortical_surfaces.py
"""
import pytest
from pathlib import Path
import logging
import shutil
import os
import trimesh

from brain_for_printing.surfgen_utils import generate_brain_surfaces

def test_generate_cortical_surfaces_basic(temp_dir, mock_vtk_available):
    """Test basic cortical surface generation workflow."""
    # Create mock FreeSurfer subject directory structure
    subjects_dir = temp_dir / "subjects"
    subject_id = "test_subject"
    subject_dir = subjects_dir / subject_id
    anat_dir = subject_dir / "anat"
    
    # Create minimal required FreeSurfer files
    os.makedirs(anat_dir, exist_ok=True)
    
    # Create mock surface files
    lh_pial = anat_dir / f"{subject_id}_hemi-L_pial.surf.gii"
    rh_pial = anat_dir / f"{subject_id}_hemi-R_pial.surf.gii"
    
    # Create simple GIFTI files
    for surf_file in [lh_pial, rh_pial]:
        with open(surf_file, "w") as f:
            f.write("mock surface data")
    
    # Test surface generation
    surfaces = generate_brain_surfaces(
        subjects_dir=str(subjects_dir),
        subject_id=subject_id,
        space="T1",
        surfaces=["pial"],
        tmp_dir=str(temp_dir / "work")
    )
    
    assert surfaces is not None
    assert isinstance(surfaces, dict)
    assert "pial_L" in surfaces
    assert "pial_R" in surfaces

def test_generate_cortical_surfaces_invalid_space(temp_dir, mock_vtk_available):
    """Test surface generation with invalid space."""
    with pytest.raises(ValueError, match="Invalid space"):
        generate_brain_surfaces(
            subjects_dir=str(temp_dir),
            subject_id="test_subject",
            space="invalid_space",
            surfaces=["pial"]
        )

def test_generate_cortical_surfaces_inflated_non_t1(temp_dir, mock_vtk_available):
    """Test that inflated surfaces are only allowed in T1 space."""
    with pytest.raises(ValueError, match="Inflated surf only T1 space"):
        generate_brain_surfaces(
            subjects_dir=str(temp_dir),
            subject_id="test_subject",
            space="MNI",
            surfaces=["inflated"]
        ) 
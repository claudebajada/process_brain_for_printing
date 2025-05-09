"""
Unit tests for aseg_utils.py
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os

from brain_for_printing.aseg_utils import convert_fs_aseg_to_t1w

@pytest.fixture
def mock_fs_aseg():
    """Create a mock FreeSurfer ASEG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs_dir = Path(tmpdir) / "sub-01" / "freesurfer" / "mri"
        fs_dir.mkdir(parents=True)
        aseg_file = fs_dir / "aseg.mgz"
        aseg_file.touch()
        yield str(Path(tmpdir))

@pytest.fixture
def mock_transform():
    """Create a mock transform file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        anat_dir = Path(tmpdir) / "sub-01" / "anat"
        anat_dir.mkdir(parents=True)
        xfm_file = anat_dir / "sub-01_from-fsnative_to-T1w_mode-image_xfm.txt"
        xfm_file.touch()
        yield str(xfm_file)

@pytest.fixture
def mock_t1w():
    """Create a mock T1w reference file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        anat_dir = Path(tmpdir) / "sub-01" / "anat"
        anat_dir.mkdir(parents=True)
        t1w_file = anat_dir / "sub-01_desc-preproc_T1w.nii.gz"
        t1w_file.touch()
        yield str(t1w_file)

def test_convert_fs_aseg_to_t1w_success(mock_fs_aseg, mock_transform, mock_t1w):
    """Test successful conversion of FreeSurfer ASEG to T1w space."""
    with patch('brain_for_printing.aseg_utils.run_cmd') as mock_run_cmd, \
         patch('brain_for_printing.aseg_utils.flexible_match') as mock_flexible_match:
        
        # Setup mocks
        mock_flexible_match.side_effect = [mock_transform, mock_t1w]
        mock_run_cmd.return_value = 0
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create intermediate files that would be created by the commands
            output_dir_path = Path(tmpdir)
            intermediate_nii = output_dir_path / "sub-01_desc-aseg_fsnative.nii.gz"
            intermediate_nii.touch()
            
            output_aseg = output_dir_path / "sub-01_desc-aseg_dseg.nii.gz"
            output_aseg.touch()
            
            # Run conversion
            result = convert_fs_aseg_to_t1w(
                subjects_dir=mock_fs_aseg,
                subject_id="sub-01",
                output_dir=tmpdir
            )
            
            # Verify result
            assert result is not None
            assert Path(result).exists()
            
            # Verify commands were called correctly
            assert mock_run_cmd.call_count == 2
            mri_convert_cmd = mock_run_cmd.call_args_list[0][0][0]
            ants_cmd = mock_run_cmd.call_args_list[1][0][0]
            
            assert mri_convert_cmd[0] == "mri_convert"
            assert ants_cmd[0] == "antsApplyTransforms"
            assert ants_cmd[1] == "-d"
            assert ants_cmd[2] == "3"
            assert ants_cmd[3] == "-i"
            assert ants_cmd[5] == "-r"
            assert ants_cmd[6] == mock_t1w
            assert ants_cmd[7] == "-t"
            assert ants_cmd[8] == mock_transform

def test_convert_fs_aseg_to_t1w_missing_aseg():
    """Test handling of missing FreeSurfer ASEG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = convert_fs_aseg_to_t1w(
            subjects_dir=tmpdir,
            subject_id="sub-01",
            output_dir=tmpdir
        )
        assert result is None

def test_convert_fs_aseg_to_t1w_missing_transform(mock_fs_aseg):
    """Test handling of missing transform file."""
    with patch('brain_for_printing.aseg_utils.flexible_match') as mock_flexible_match:
        mock_flexible_match.side_effect = FileNotFoundError("Transform not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_fs_aseg_to_t1w(
                subjects_dir=mock_fs_aseg,
                subject_id="sub-01",
                output_dir=tmpdir
            )
            assert result is None

def test_convert_fs_aseg_to_t1w_with_session(mock_fs_aseg, mock_transform, mock_t1w):
    """Test conversion with session ID."""
    with patch('brain_for_printing.aseg_utils.run_cmd') as mock_run_cmd, \
         patch('brain_for_printing.aseg_utils.flexible_match') as mock_flexible_match:
        
        # Setup mocks
        mock_flexible_match.side_effect = [mock_transform, mock_t1w]
        mock_run_cmd.return_value = 0
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create intermediate files that would be created by the commands
            output_dir_path = Path(tmpdir)
            intermediate_nii = output_dir_path / "sub-01_desc-aseg_fsnative.nii.gz"
            intermediate_nii.touch()
            
            output_aseg = output_dir_path / "sub-01_ses-01_desc-aseg_dseg.nii.gz"
            output_aseg.touch()
            
            # Run conversion with session
            result = convert_fs_aseg_to_t1w(
                subjects_dir=mock_fs_aseg,
                subject_id="sub-01",
                output_dir=tmpdir,
                session="01"
            )
            
            # Verify result
            assert result is not None
            assert "ses-01" in result
            assert Path(result).exists() 
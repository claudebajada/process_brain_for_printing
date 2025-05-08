"""
pytest configuration and common fixtures.
"""
import os
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_vtk_file(temp_dir):
    """Create a sample VTK file for testing."""
    vtk_path = Path(temp_dir) / "sample.vtk"
    # Create a simple VTK file with a cube
    vtk_content = """# vtk DataFile Version 3.0
Cube example
ASCII
DATASET POLYDATA
POINTS 8 float
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
POLYGONS 6 30
4 0 1 2 3
4 4 5 6 7
4 0 1 5 4
4 2 3 7 6
4 0 3 7 4
4 1 2 6 5
"""
    vtk_path.write_text(vtk_content)
    return str(vtk_path)

@pytest.fixture
def mock_vtk_available(monkeypatch):
    """Mock VTK availability."""
    monkeypatch.setattr("brain_for_printing.five_tt_utils.VTK_AVAILABLE", True)
    return True

@pytest.fixture
def mock_vtk_unavailable(monkeypatch):
    """Mock VTK unavailability."""
    monkeypatch.setattr("brain_for_printing.five_tt_utils.VTK_AVAILABLE", False)
    return False 
# CLAUDE.md — brain_for_printing

This file provides context for AI assistants working in this repository.

## Project Overview

`brain_for_printing` is a Python package for generating 3D-printable brain surfaces from neuroimaging (MRI) data. It extracts cortical, subcortical, brainstem, and cerebellar surfaces from FreeSurfer derivatives, applies coordinate space transforms (T1 native, MNI, or target subject), and outputs mesh files (STL/GIFTI) ready for 3D printing — including multi-material slab components and parametric vertex coloring.

- **Author:** Claude J. Bajada
- **Version:** 0.1.0
- **License:** MIT
- **Python:** ≥3.7

---

## Repository Structure

```
process_brain_for_printing/
├── brain_for_printing/          # Main Python package
│   ├── __init__.py              # Package version (0.1.0)
│   ├── constants.py             # FreeSurfer ASEG labels, CLI choice constants
│   ├── surfaces.py              # Compatibility re-export shim
│   ├── log_utils.py             # Logger setup, JSON run-log writer
│   ├── io_utils.py              # run_cmd(), flexible_match(), subject validation
│   ├── config_utils.py          # PRESETS dict and preset parsing
│   ├── surfgen_utils.py         # Cortical + brain mask surface generation
│   ├── aseg_utils.py            # ASEG-based structure extraction (brainstem, cerebellum, CC)
│   ├── five_tt_utils.py         # 5ttgen workflow + VTK mesh loading
│   ├── mesh_utils.py            # GIFTI↔trimesh conversion, voxelization, remeshing
│   ├── volumetric_utils.py      # mrgrid regridding, mesh2voxel, numpy volume ops
│   ├── warp_utils.py            # ANTs→MRtrix warp field generation
│   ├── color_utils.py           # Parameter map sampling + vertex coloring
│   ├── cli_cortical_surfaces.py # CLI: cortical + ASEG + 5ttgen surfaces in any space
│   ├── cli_color.py             # CLI: mesh coloring (direct mesh or preset mode)
│   └── cli_multi_material_slab_components.py  # CLI: AC-PC aligned multi-material slabs
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── unit/
│   │   ├── test_aseg_utils.py
│   │   └── test_five_tt_utils.py
│   └── integration/
│       ├── test_cortical_surfaces.py
│       └── test_five_tt_workflow.py
├── pyproject.toml               # Build config, dependencies, entry points
├── pytest.ini                   # Pytest settings (markers, coverage, log level)
└── README.md                    # User-facing documentation
```

---

## Development Setup

```bash
# Install in editable mode (registers all CLI entry points)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Code Quality

```bash
black --line-length 100 brain_for_printing/   # Format
isort --profile black brain_for_printing/     # Sort imports
flake8 brain_for_printing/                    # Lint
mypy brain_for_printing/                      # Type check
```

### Running Tests

```bash
pytest                                         # All tests
pytest tests/unit/                             # Unit tests only
pytest tests/integration/                      # Integration tests only
pytest -v --cov=brain_for_printing --cov-report=html  # With coverage
pytest -m "not slow"                           # Skip slow tests
pytest -m "not vtk"                            # Skip VTK-dependent tests
```

Test markers (defined in `pytest.ini`): `unit`, `integration`, `slow`, `vtk`.

---

## CLI Entry Points

| Command | Source | Description |
|---|---|---|
| `brain_for_printing_cortical_surfaces` | `cli_cortical_surfaces:main` | Extract cortical, ASEG-derived, and 5ttgen surfaces in T1/MNI/target space |
| `brain_for_printing_multi_material_slabs` | `cli_multi_material_slab_components:main` | Generate AC-PC-aligned multi-material slab components |
| `brain_for_printing_slabs` | `cli_multi_material_slab_components:main` | Alias for the above |
| `brain_for_printing_color` | `cli_color:main` | Color meshes from a parametric volume (direct or preset mode) |

> **Note:** `pyproject.toml` lists two additional entry points (`brain_for_printing_brain_mask_surface` and `brain_for_printing_overlay`) that reference non-existent source files (`cli_brain_mask_surface`, `cli_overlay`). Brain mask surface generation is handled via the `brain_mask_surface` preset in `cli_cortical_surfaces`.

### Subcommand Pattern

`cli_cortical_surfaces` and `cli_color` use argparse subcommands:
- `brain_for_printing_cortical_surfaces preset` — use a named preset
- `brain_for_printing_cortical_surfaces direct` — specify surfaces directly
- `brain_for_printing_color preset` — color a preset
- `brain_for_printing_color direct` — color an existing mesh file

---

## Architecture & Key Patterns

### Module Layers

```
CLI Layer          cli_*.py              Argument parsing, signal handling, run-log writing
Workflow Layer     surfgen_utils.py      Surface generation orchestration
                   five_tt_utils.py      5ttgen pipeline
                   volumetric_utils.py   Volume operations
Integration Layer  warp_utils.py         ANTs↔MRtrix coordinate transforms
                   io_utils.py           External command execution, file discovery
                   aseg_utils.py         FreeSurfer ASEG handling
Core Layer         mesh_utils.py         GIFTI/trimesh/VTK conversions
                   color_utils.py        Vertex coloring from parameter maps
Config Layer       constants.py          FreeSurfer label sets
                   config_utils.py       Preset definitions
                   log_utils.py          Logger + JSON audit trail
```

### Key Design Patterns

- **External command execution:** All shell calls go through `io_utils.run_cmd()`. Never use `subprocess` directly.
- **Flexible file discovery:** Use `io_utils.flexible_match()` to locate BIDS-named files by partial pattern.
- **Logging:** Every module defines `L = logging.getLogger(__name__)` at the top. Use `-v` flag in CLI for verbose output.
- **Run-logs:** Each CLI invocation writes a timestamped JSON log via `log_utils.write_run_log()` for provenance tracking.
- **Presets:** Surface combinations are defined in `config_utils.PRESETS`. Add new presets there rather than hardcoding in CLIs.
- **Graceful exit:** All CLIs install SIGINT/SIGTERM handlers. First Ctrl+C prints a message; second force-exits.
- **Space-agnostic transforms:** Supports T1 native, MNI, or arbitrary target subject space via ANTs H5 transforms converted to MRtrix warp fields in `warp_utils.py`.

### Defined Presets (`config_utils.PRESETS`)

```python
"pial_brain"      -> lh-pial, rh-pial, corpus_callosum, cerebellum, brainstem
"white_brain"     -> lh-white, rh-white, corpus_callosum, cerebellum_wm, brainstem
"mid_brain"       -> lh-mid, rh-mid, corpus_callosum, cerebellum, brainstem
"cortical_pial"   -> lh-pial, corpus_callosum, rh-pial
"cortical_white"  -> lh-white, corpus_callosum, rh-white
"cortical_mid"    -> lh-mid, corpus_callosum, rh-mid
"brain_mask_surface" -> brain_mask_indicator
```

---

## Code Conventions

### Style
- **Formatter:** `black` with `--line-length 100`
- **Import sorting:** `isort` with `--profile black`
- **Linter:** `flake8`
- **Type checking:** `mypy` (partial annotations; not all functions are fully annotated)

### Naming
- Modules: `snake_case`, suffixed `_utils.py` for helpers or `cli_` for entry points
- Functions: `snake_case`; private helpers prefixed `_`
- Constants: `UPPER_SNAKE_CASE`

### Logging
```python
import logging
L = logging.getLogger(__name__)  # standard pattern in every module

L.info("Processing subject %s", subject_id)
L.debug("Running command: %s", cmd)
L.warning("File not found: %s", path)
```

### Testing
- Mock external binaries (FreeSurfer, ANTs, MRtrix) — do not rely on them being installed in tests
- Fixtures in `tests/conftest.py`: `tmp_dir`, `sample_vtk_file`, `mock_vtk_available`
- Tag tests with appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.vtk`

---

## External Binary Requirements

The package requires several external neuroimaging tools at runtime (not Python deps):

| Tool | Purpose |
|---|---|
| **FreeSurfer** | ASEG segmentation, `.mgz` → `.nii.gz` conversion, cortical surface files |
| **ANTs** | T1↔MNI warp fields, subject-to-subject transforms |
| **MRtrix3** | Warp application, `mesh2voxel`, `voxel2mesh`, `5ttgen`, `mrgrid` |
| **FSL** | AC-PC alignment (`robustfov`, `flirt`), registration |

These are called via `io_utils.run_cmd()`. In tests, mock them with `unittest.mock.patch`.

---

## FreeSurfer Label Constants (`constants.py`)

Key label groups used in ASEG extraction:
- `BRAINSTEM_LABELS` — label 16 plus 30 related brainstem structure labels
- `CEREBELLUM_CORTEX_LABELS` — labels 8, 47
- `CEREBELLUM_WM_LABELS` — labels 7, 46
- `CORPUS_CALLOSUM_LABELS` — labels 251–255

VTK-derived structure names for 5ttgen output are also defined here (e.g., `L_Thal`, `R_Hipp`, `3rd-Ventricle`).

---

## Common Gotchas

1. **Missing CLI files:** `cli_brain_mask_surface.py` and `cli_overlay.py` are referenced in `pyproject.toml` but do not exist. Do not add new entry points there without creating the corresponding module.

2. **VTK availability:** VTK loading in `five_tt_utils.py` has a fallback chain (`vtkPolyDataReader` → `vtkGenericDataObjectReader`). Mark VTK-dependent tests with `@pytest.mark.vtk`.

3. **ANTs H5 transforms:** Must be converted to MRtrix `.mif` warp fields before use — handled by `warp_utils.generate_warp_field()`.

4. **AC-PC alignment:** The multi-material slab CLI operates entirely in AC-PC space. All input meshes must be transformed before slicing.

5. **Editable install required:** CLI entry points only work after `pip install -e .`. Raw `python brain_for_printing/cli_*.py` invocation is not supported.

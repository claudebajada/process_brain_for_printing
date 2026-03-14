# CLAUDE.md — brain_for_printing

## Project Overview

`brain_for_printing` is a Python CLI package that generates 3D-printable brain surface meshes from BIDS-formatted neuroimaging data. It extracts cortical surfaces (pial/white/mid), subcortical structures, brainstem, cerebellum, ventricles, and corpus callosum from FreeSurfer derivatives, supports optional coloring from parametric volumes (e.g., fMRI contrast maps), and can produce multi-material AC-PC-aligned slab components for advanced multi-material 3D printing workflows. Space transformations between T1 native, MNI, and target-subject spaces are handled via ANTs + MRtrix3.

---

## Repo Layout

```
process_brain_for_printing/
├── brain_for_printing/              # Main Python package (v0.1.0)
│   ├── __init__.py                  # Version string
│   ├── constants.py                 # FreeSurfer ASEG label lists; CLI choice strings
│   ├── config_utils.py              # Surface presets dict + preset parser
│   ├── log_utils.py                 # Logger factory; JSON provenance log writer
│   ├── io_utils.py                  # run_cmd(), require_cmd(), temp_dir(), flexible_match(), export_surfaces()
│   ├── mesh_utils.py                # gifti_to_trimesh(), volume_to_gifti(), voxel_remesh_and_repair()
│   ├── volumetric_utils.py          # MRtrix wrappers (mrgrid, mesh2voxel, voxel2mesh); numpy volume math
│   ├── color_utils.py               # project_param_to_surface(), copy_vertex_colors()
│   ├── warp_utils.py                # create_mrtrix_warp(), warp_gifti_vertices() (ANTs ↔ MRtrix)
│   ├── aseg_utils.py                # extract_structure_surface(), convert_fs_aseg_to_t1w()
│   ├── five_tt_utils.py             # run_5ttgen_hsvs_save_temp_bids(), VTK mesh loaders
│   ├── surfgen_utils.py             # generate_brain_surfaces(), generate_single_brain_mask_surface()
│   ├── surfaces.py                  # Compatibility re-export layer (do not add new logic here)
│   ├── cli_cortical_surfaces.py     # CLI: cortical + ASEG + 5ttgen surfaces
│   ├── cli_color.py                 # CLI: mesh coloring (direct or preset mode)
│   └── cli_multi_material_slab_components.py  # CLI: multi-material AC-PC slab components
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures (temp_dir, sample VTK, mock VTK)
│   ├── unit/
│   │   ├── test_aseg_utils.py
│   │   └── test_five_tt_utils.py
│   └── integration/
│       ├── test_cortical_surfaces.py
│       └── test_five_tt_workflow.py
├── pyproject.toml                   # Dependencies, entry points, tool config (black/isort/mypy)
├── pytest.ini                       # Test discovery + coverage settings
└── README.md                        # User-facing quick-start and full CLI reference
```

---

## Development Commands

```bash
# Install (editable)
pip install -e .

# Run all tests with coverage
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Format
black brain_for_printing/
isort brain_for_printing/

# Lint / type-check
flake8 brain_for_printing/
mypy brain_for_printing/
```

Code style: **Black** with `line-length = 100`, **isort** in Black-compatible mode. mypy is configured in strict mode (`disallow_untyped_defs = true`).

---

## CLI Entry Points

| Command | Module | Purpose |
|---|---|---|
| `brain_for_printing_cortical_surfaces` | `cli_cortical_surfaces.py` | Extract cortical (LH/RH pial/white/mid), ASEG (brainstem, cerebellum, CC), and 5ttgen (subcortical, ventricles) surfaces. Supports presets and custom selections. |
| `brain_for_printing_color` | `cli_color.py` | Color meshes from parameter volumes. Subcommands: `direct` (existing mesh) or `preset` (generate + color). |
| `brain_for_printing_multi_material_slabs` | `cli_multi_material_slab_components.py` | Generate distinct non-overlapping STL components (OuterCSF, GM, WM, Ventricles, SGM) in AC-PC space. |
| `brain_for_printing_brain_mask_surface` | `cli_cortical_surfaces.py` | Surface from a binary brain mask with optional inflation/smoothing. |
| `brain_for_printing_overlay` | *(deprecated/not yet implemented)* | — |

---

## Architecture & Processing Pipeline

```
BIDS Dataset (T1w + FreeSurfer derivatives)
        │
        ▼
[surfgen_utils] ──── cortical surfaces (GIFTI → trimesh via mesh_utils)
        │        ──── ASEG structures  (aseg_utils → volumetric_utils → mesh_utils)
        │        ──── 5ttgen surfaces  (five_tt_utils → VTK readers)
        │
        ▼
[warp_utils] — optional space transform (T1 native ↔ MNI ↔ target subject)
        │        uses: ANTs antsApplyTransforms + MRtrix warpinit/mrcat
        │
        ▼
[color_utils] — optional vertex coloring from parametric volume
        │
        ▼
[io_utils] — export STL/OBJ files + JSON provenance log
```

### Multi-Material Slab Pipeline (cli_multi_material_slab_components.py)
1. AC-PC alignment via FSL FLIRT
2. Resample to high-res isotropic voxels (`mrgrid`)
3. Generate parent surfaces (cortical + ASEG + 5ttgen)
4. Voxelize meshes (`mesh2voxel`)
5. Volumetric slicing to the requested slab
6. Material layer construction with numpy boolean ops
7. Mesh extraction (`voxel2mesh`) per material → STL output

---

## Key Module Responsibilities

| Module | Key Functions | Notes |
|---|---|---|
| `io_utils.py` | `run_cmd()`, `require_cmd()`, `temp_dir()`, `flexible_match()`, `export_surfaces()` | All subprocess calls go through `run_cmd()`. Always use `require_cmds()` at CLI entry to validate external tools. |
| `mesh_utils.py` | `gifti_to_trimesh()`, `volume_to_gifti()`, `voxel_remesh_and_repair()` | GIFTI ↔ trimesh conversion; marching cubes for volumetric surfaces. |
| `volumetric_utils.py` | `regrid_to_resolution()`, `mesh_to_partial_volume()`, `vol_union_numpy()` | MRtrix wrappers + numpy volume algebra. |
| `warp_utils.py` | `create_mrtrix_warp()`, `warp_gifti_vertices()` | Builds 4D MRtrix warp field from ANTs transforms; applies to GIFTI vertex coords. |
| `aseg_utils.py` | `extract_structure_surface()`, `convert_fs_aseg_to_t1w()` | FreeSurfer ASEG label → binary mask → surface. |
| `five_tt_utils.py` | `run_5ttgen_hsvs_save_temp_bids()`, `load_subcortical_and_ventricle_meshes()` | Runs MRtrix `5ttgen hsvs`; loads resulting VTK meshes with fallback readers. |
| `surfgen_utils.py` | `generate_brain_surfaces()`, `generate_single_brain_mask_surface()` | Top-level surface generation; orchestrates all other utils. |
| `color_utils.py` | `project_param_to_surface()`, `copy_vertex_colors()` | Samples parametric NIfTI at mesh vertices; discretizes into color bins. |
| `config_utils.py` | `PRESETS`, `parse_preset()` | Predefined surface combinations (e.g., `pial_brain`, `mid_brain`). Extend `PRESETS` dict to add new presets. |
| `constants.py` | `BRAINSTEM_LABEL`, `CEREBELLUM_*_LABELS`, `CORPUS_CALLOSUM_LABELS`, `CORTICAL_TYPES` | FreeSurfer ASEG integer labels and valid CLI choice strings. |
| `log_utils.py` | `get_logger()`, `write_log()` | JSON run-log captures git hash + system info. Every module should call `L = logging.getLogger(__name__)`. |
| `surfaces.py` | re-exports only | Backward-compatibility shim. Do not add new logic here. |

---

## External Tool Requirements

All tools must be on `PATH`. Use `require_cmds()` from `io_utils` at the top of each CLI.

| Tool | Commands Used | Purpose |
|---|---|---|
| **FreeSurfer** | `mri_convert`, `mri_binarize` | `.mgz` conversion, ASEG binarization |
| **ANTs** | `antsApplyTransforms` | Space-to-space transforms (T1 ↔ MNI ↔ subject) |
| **MRtrix3** | `warpinit`, `mrcat`, `mrgrid`, `mesh2voxel`, `voxel2mesh`, `5ttgen` | Warp fields, resampling, surface ↔ volume, tissue segmentation |
| **FSL** | `flirt`, `robustfov`, `fslroi` | AC-PC alignment, FOV cropping |
| **OpenSCAD / Blender** *(optional)* | — | trimesh boolean operation backends |

---

## Code Conventions

- **Line length:** 100 characters (Black).
- **Imports:** isort with Black profile; stdlib → third-party → local (relative imports within package).
- **Logging:** Every module declares `L = logging.getLogger(__name__)` at module level. Do not use `print()` for status messages.
- **Subprocess:** Always use `io_utils.run_cmd()`. Never call `subprocess` directly in feature code.
- **Temp files:** Use the `temp_dir()` context manager from `io_utils`. Pass `cleanup=False` only for debugging.
- **Constants:** ASEG label integers and valid CLI strings live in `constants.py`. Add new labels there.
- **Type hints:** Required on all new functions (mypy strict). Use `Optional[X]` not `X | None` (Python 3.7 compat).
- **Docstrings:** Google-style (`Args:`, `Returns:`, `Raises:`).
- **Naming:** `snake_case` functions/variables, `UPPER_CASE` module-level constants, `PascalCase` classes (rare).

---

## Testing

```bash
pytest                        # All tests + coverage
pytest -m unit                # Unit tests only
pytest -m integration         # Integration tests only
pytest -m "not slow"          # Skip slow tests
pytest -m "not vtk"           # Skip tests requiring VTK
```

### Test Markers (defined in pytest.ini)
| Marker | Use For |
|---|---|
| `unit` | Fast, no external tools, fully mocked |
| `integration` | Requires external tools or real file I/O |
| `slow` | Long-running tests |
| `vtk` | Tests requiring VTK installation |

### Fixtures (tests/conftest.py)
- `temp_dir` — provides a temporary directory, cleaned up after each test
- `sample_vtk_file` — minimal mock VTK file for testing VTK readers
- `mock_vtk_available` — patches VTK availability flag

### Mocking Pattern
External tool calls (`run_cmd`) are patched in unit tests. Integration tests should use real (or minimal) BIDS data where possible. Do not write unit tests that shell out to FreeSurfer/ANTs/MRtrix.

---

## Branch Strategy

- `main` — stable releases
- `dev` — integration branch for features
- Feature branches: descriptive names, merged via PR into `dev`

Commit message style: `Fix:`, `Feat:`, `Refactor:`, `Note:` prefixes (see git log).

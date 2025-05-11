# Brain For Printing

Generate **3D‑printable brain surfaces** (cortical, subcortical, brainstem, etc.) in T1, MNI, or target subject space, color them from parametric volumes, or slice them into stackable slabs – all from the command line.

The core is pure‑Python (`trimesh`, `nibabel`, `scikit-image`) while heavy lifting is delegated to familiar neuro‑tools (FreeSurfer, FSL, ANTs, MRtrix3, 5ttgen). Every CLI writes a timestamped JSON *run‑log* for provenance and supports structured console logging via `-v/--verbose`.

---

## Key features

| Area                 | Highlights                                                                                                                                                                 |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Surface Extraction** | • LH/RH pial, white, **or** mid cortical surfaces. <br>• Extraction of ASEG-derived structures (brainstem, cerebellum, corpus callosum) with optional hole‑fill & smoothing.<br>• Integration of 5ttgen-derived VTK meshes (subcortical gray matter, ventricles) for T1 space.<br>• Output in T1 native, MNI template, or target subject space (subject-to-subject warping via MNI).<br>• Automatic ANTs→MRtrix 4‑D warp field generation for space transformations. |
| **Coloring** | • Sample NIfTI parameter maps per‑vertex in source (T1) or target space.<br>• Thresholding & discrete color bins.<br>• Works on existing STL/OBJ/GIFTI **or** freshly generated surfaces using presets.<br>• Flexible sampling surface selection (e.g., white, pial, mid) for cortical targets. |
| **Mesh Operations** | • Voxel remeshing and repair helper for creating watertight meshes.<br>• Slab slicing with per‑slab bounding‑box padding for alignment of multiple meshes.                 |

---

## Installation

```bash
# 1. Clone
git clone [https://github.com/claudebajada/process_brain_for_printing.git](https://github.com/claudebajada/process_brain_for_printing.git)
cd process_brain_for_printing

# 2. Editable install + entry‑points
pip install -e .
```

> **Python ≥3.8** recommended. All Python dependencies (see `requirements.txt`) are pulled in automatically. VTK is required for handling some subcortical/ventricular meshes.

---

## External Binaries

Depending on the sub‑command and features you use, you may need some (or all) of:

| Tool                                       | Needed for                                                                                                  | URL                                            |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| **FreeSurfer** (`mri_convert`, `mri_binarize`) | ASEG handling, mask generation, conversion of `.mgz` files.                                                 | <https://surfer.nmr.mgh.harvard.edu>           |
| **ANTs** (`antsApplyTransforms`)             | T1⇄MNI warps, subject-to-subject warps, applying transformations.                                           | <https://github.com/ANTsX/ANTs>                |
| **MRtrix3** (`warpinit`, `mrcat`)          | Generation and manipulation of 4‑D warp fields for transformations.                                         | <https://www.mrtrix.org>                       |
| **MRtrix3** (`5ttgen`)                     | Generating 5-tissue-type segmentations and associated VTK meshes (subcortical, ventricles).                   | <https://www.mrtrix.org>                       |
| **OpenSCAD** or **Blender** | Optional: fast boolean backend for `trimesh` operations (e.g., slab slicing).                               | <https://openscad.org>, <https://blender.org> |

CLIs abort early with a helpful message if a required binary is missing for the requested operation.

---

## Command‑Line Tools

| Command                                  | Purpose                                                                                                                                                                  |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `brain_for_printing_cortical_surfaces` | Extract cortical (LH/RH), ASEG-derived (cerebellum, brainstem, CC), and 5ttgen-derived (subcortical, ventricles) surfaces. Outputs STLs. Supports presets and custom selections. |
| `brain_for_printing_color`               | Color meshes. **`direct`** mode colors an existing mesh file. **`preset`** mode generates surfaces using a preset, then selectively colors and combines them.           |
| `brain_for_printing_slab_slices`         | Slice one or more meshes into aligned volumetric slabs.                                                                                                                    |
| `brain_for_printing_brain_mask_surface`  | Create a surface mesh from a binary brain mask (T1 or MNI space), with optional inflation and smoothing.                                                                  |

Run any command with `-h/--help` for a full argument reference.

---

## Quick‑Start Examples

### 1. Surface Generation (`brain_for_printing_cortical_surfaces`)

**Using a preset for pial surfaces, brainstem, cerebellum, and corpus callosum in T1 space:**
```bash
brain_for_printing_cortical_surfaces preset \
  --subjects_dir /path/to/bids/derivatives \
  --subject_id sub-01 \
  --name pial_brain \
  --space T1 \
  --output_dir ./models_t1_preset \
  --split_outputs \
  -v
```

**Custom selection including cortical, ASEG, and 5ttgen-derived (VTK) subcortical structures in T1 space:**
```bash
brain_for_printing_cortical_surfaces custom \
  --subjects_dir /path/to/bids/derivatives \
  --subject_id sub-01 \
  --cortical-surfaces lh-pial rh-pial \
  --cbm-bs-cc brainstem cerebellum \
  --subcortical-gray L_Thal R_Thal all # 'all' for all available SGM, or specific names
  --space T1 \
  --output_dir ./models_t1_custom \
  --work_dir /path/to/work_directory_for_5ttgen # Optional: specify base for 5ttgen temp/persistent files
  --split_outputs \
  -v
```

*Flags of note for `brain_for_printing_cortical_surfaces`*:
* Mode: `preset --name <preset_name>` or `custom`
* `--cortical-surfaces`: e.g., `pial`, `lh-white`, `rh-mid` (for custom mode)
* `--cbm-bs-cc`: e.g., `brainstem`, `cerebellum`, `corpus_callosum` (for custom mode)
* `--subcortical-gray`: e.g., `L_Hipp`, `R_Amyg`, `all` (for custom mode, requires 5ttgen data)
* `--ventricular-system`: e.g., `3rd-Ventricle`, `all` (for custom mode, requires 5ttgen data)
* `--space T1|MNI|sub-XX`: Target output space.
* `--split_outputs`: Export each generated mesh as a separate file.
* `--no-fill-structures` / `--no-smooth-structures`: Selectively skip hole filling or smoothing for ASEG structures.
* `--work_dir`: Base directory for intermediate files, especially for 5ttgen processing.
* `--no_clean`: Keep temporary folders.
* `-v`: Verbose logging.

### 2. Generate & Color Surfaces (`brain_for_printing_color preset`)

**Color a "pial_brain" preset in MNI space, sampling from the white matter surface:**
```bash
brain_for_printing_color preset \
  --subjects_dir /path/to/bids/derivatives \
  --subject_id sub-01 \
  --preset pial_brain \
  --space MNI \
  --color_in target \
  --param_map /path/to/tstat_in_MNI.nii.gz \
  --color_sampling_surf white \
  --num_colors 9 \
  --output_dir ./coloured_models_mni \
  -v
```

**Color in source T1 space, then warp surfaces to target subject `sub-02`:**
```bash
brain_for_printing_color preset \
  --subjects_dir /path/to/bids/derivatives \
  --subject_id sub-01 \
  --preset mid_brain \
  --space sub-02 \
  --color_in source \
  --param_map /path/to/tstat_in_T1_sub-01.nii.gz \
  --color_sampling_surf mid \
  --num_colors 6 \
  --output_dir ./coloured_models_sub02_from_sub01 \
  -v
```

*Flags of note for `brain_for_printing_color preset`*:
* `--preset <name>`: e.g., `pial_brain`, `white_brain`, `mid_brain`.
* `--space T1|MNI|sub-XX`: Space for final surface generation.
* `--color_in source|target`: Space in which coloring is performed (`source` is subject's T1, `target` is the final `--space`).
* `--param_map <path_to_nifti>`: Parameter map for coloring.
* `--color_sampling_surf white|pial|mid`: Cortical surface type used to sample colors for other cortical surfaces.
* `--num_colors <N>`: Number of discrete color bins.
* `--param_threshold <value>`: Optional threshold for parameter map.

### 3. Color an Existing Mesh Directly (`brain_for_printing_color direct`)

```bash
brain_for_printing_color direct \
  --in_mesh ./input_brain.obj \
  --param_map ./perfusion_map.nii.gz \
  --out_obj ./brain_coloured.obj \
  --num_colors 7
```

### 4. Slice a Brain STL into 10mm Axial Slabs

```bash
brain_for_printing_slab_slices \
  --in_meshes sub-01_T1_pial_brain.stl \
  --orientation axial \
  --thickness 10.0 \
  --out_dir ./slabs_axial
```

---

## Logging & Reproducibility

* Real‑time logging is controlled by `-v/--verbose`.
* Every run writes a JSON log (`*_log_<timestamp>.json`) capturing inputs, parameters, steps, warnings, and outputs.
* Temporary folders are auto‑removed unless you pass `--no_clean`.

---

## Testing

The project includes a suite of unit and integration tests using `pytest`.
* Run tests with `pytest` in the root directory.
* Test coverage reports can be generated (see `pytest.ini`).
* Current tests cover core utilities, argument parsing, and some workflow logic, with ongoing efforts to expand coverage, especially with diverse dummy data.

---

## Roadmap

* **Enhanced Test Coverage**: Continue expanding unit and integration tests with more diverse dummy datasets and edge cases.
* **General Mesh Boolean Operations CLI**: Potentially add a dedicated CLI tool for more general boolean operations on meshes.
* **Configuration for External Tools**: Explore options for specifying paths to external tool binaries if not in PATH.

---

## License

MIT © Claude J. Bajada and contributors

---

## Acknowledgements

* Neuroimaging software: FSL, FreeSurfer, ANTs, MRtrix3.
* Python libraries: numpy, nibabel, scipy, scikit‑image, trimesh, vtk.
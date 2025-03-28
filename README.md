# Brain For Printing

**Brain For Printing** is a Python package that generates 3D-printable brain surfaces (pial + brainstem) in T1/MNI space, optionally colors them using a parameter map, and can create volumetric “slab” slices for stackable 3D prints. It relies on common neuroimaging tools (FSL, FreeSurfer, ANTs, etc.) plus Python libraries (`nibabel`, `numpy`, `trimesh`, etc.) to perform these operations.

---

## Features

1. **Extract & Warp Surfaces**:  
   - **T1 or MNI space** (LH/RH pial or white, optional brainstem).  
2. **Color Surfaces**:  
   - **Param map projection** (nearest or linear interpolation).  
   - **Mid-thickness sampling** to color pial surfaces.  
   - **Overlay segmentation + param maps**.  
3. **Brainstem Extraction**:  
   - Binarize specific labels in a Freesurfer aseg for T1 or MNI.  
4. **Volumetric Slicing**:  
   - Slice a 3D STL into multiple slabs along axial/coronal/sagittal orientation.  
   - Optionally subdivide each slab’s mesh for denser vertex sampling.  
5. **Hollowing Ventricles**:  
   - Subtract ventricles from a brain mesh using fMRIPrep segmentation.  

---

## Table of Contents

- [Installation](#installation)  
- [Dependencies](#dependencies)  
- [Usage](#usage)  
  - [1. Cortical Surfaces (T1 or MNI)](#1-cortical-surfaces-t1-or-mni)  
  - [2. Brainstem Extraction](#2-brainstem-extraction)  
  - [3. Coloring Surfaces](#3-coloring-surfaces)  
  - [4. Volumetric Slicing](#4-volumetric-slicing)
  - [5. Hollow Ventricles](#5-hollow-ventricles)
- [Testing](#testing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  

---

## Installation

1. Clone or download this repository:

   ```bash
   git clone https://github.com/claudebajada/process_brain_for_printing.git
   cd process_brain_for_printing
   ```

2. Install in **editable mode**:

   ```bash
   pip install -e .
   ```

   This will install the `brain_for_printing` Python library plus register multiple CLI commands (see [Usage](#usage)).

---

## Dependencies

### Python Packages

- [Python 3.7+](https://www.python.org/)  
- [numpy](https://numpy.org/)  
- [nibabel](https://nipy.org/nibabel/)  
- [scipy](https://scipy.org/)  
- [scikit-image](https://scikit-image.org/)  
- [matplotlib](https://matplotlib.org/)  
- [trimesh](https://github.com/mikedh/trimesh)

These are automatically installed via `pip` if not already present.

### External Neuroimaging Tools

Depending on the functionality you want, you may need:

- **Freesurfer** (e.g., for `mri_convert`, `mri_binarize`)  
- **FSL** (e.g., for `fslmaths`)  
- **ANTs** (e.g., `antsApplyTransforms`)  
- **MRtrix** (e.g., `warpinit`, `mrcat`)

Make sure these commands are on your system’s `$PATH` before running the scripts that call them.

### Optional Tools

- **OpenSCAD** or **Blender** if you are doing volumetric slicing with boolean operations (trimesh may invoke one of these as a backend).

---

## Usage

Once installed, you will have several CLI commands:

- `brain_for_printing_cortical_surfaces`
- `brain_for_printing_brainstem`
- `brain_for_printing_color`
- `brain_for_printing_slab_slices`
- `brain_for_printing_hollow_ventricles`
- `brain_for_printing_overlay`
- `brain_for_printing_brain_mask_surface`

---

### 1) Cortical Surfaces (T1 or MNI)

Generate cortical surfaces (LH, RH, and optionally brainstem) in either **T1** or **MNI** space.

```bash
brain_for_printing_cortical_surfaces \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --space T1 \
  --output_dir /path/to/output \
  --use_white \
  --no_brainstem \
  --split_hemis
```

**Arguments**:
- `--space T1` (default) or `--space MNI`: Output in native or MNI space.
- `--use_white`: Use white matter surfaces instead of the default pial.
- `--no_brainstem`: Skip extracting the brainstem surface.
- `--split_hemis`: Export LH, RH, and brainstem as separate STL files. If not set, a combined mesh is exported.
- `--no_fill` / `--no_smooth`: Skip hole-filling or smoothing for the brainstem mesh.
- `--output_dir`: Where to save STL files and the process log.
- `--verbose`: Print extra information.
- `--no_clean`: Retain temporary folder.

Use `brain_for_printing_color` or `brain_for_printing_overlay` to color these surfaces after generation.

---

### 2) Brainstem Extraction

Extract only the brainstem (or your custom label set) in T1 or MNI space:

```bash
brain_for_printing_brainstem \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --space T1 \
  --output_dir /path/to/output
```

---

### 3) Coloring Surfaces

Color an existing mesh using a parametric volume (NIfTI). Works on STL, OBJ, or GIFTI.

```bash
brain_for_printing_color \
  --in_mesh some_brain_mesh.stl \
  --param_map volume_in_T1.nii.gz \
  --out_obj colored_mesh.obj \
  --num_colors 6
```

To use anatomical segmentation + param overlay, use:

```bash
brain_for_printing_overlay \
  --in_mesh mesh.obj \
  --seg segmentation.nii.gz \
  --param param_map.nii.gz \
  --out_obj overlay_colored.obj
```

---

### 4) Volumetric Slicing

Slice a 3D mesh into multiple slabs for stackable printing.

```bash
brain_for_printing_slab_slices \
  --in_mesh sub-01_T1_brain.stl \
  --orientation axial \
  --thickness 10 \
  --subdivide_max_edge 2.0 \
  --out_dir slabs_out
```

---

### 5) Hollow Ventricles

Hollow out the ventricular system from a brain mesh using fMRIPrep outputs:

```bash
brain_for_printing_hollow_ventricles \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --in_mesh sub-01_T1_brain.stl \
  --space T1 \
  --output sub-01_T1_hollowed.stl
```

---

## License

This project is licensed under the terms of the **MIT License**. 

---

## Acknowledgments

- **Neuroimaging Tools**: FSL, FreeSurfer, ANTs, MRtrix, etc.  
- **Python Libraries**: nibabel, numpy, trimesh, etc.  

---

*Thank you for using and contributing to Brain For Printing! Please [open an issue](https://github.com/claudebajada/process_brain_for_printing/issues) or reach out if you have any questions or suggestions.*

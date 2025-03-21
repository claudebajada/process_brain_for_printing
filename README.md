# Brain For Printing

**Brain For Printing** is a Python package that generates 3D-printable brain surfaces (pial + brainstem) in T1/MNI space, optionally colors them using a parameter map, and can create volumetric “slab” slices for stackable 3D prints. It relies on common neuroimaging tools (FSL, FreeSurfer, ANTs, etc.) plus Python libraries (`nibabel`, `numpy`, `trimesh`, etc.) to perform these operations.

---

## Features

1. **Extract & Warp Surfaces**:  
   - **T1-space** (LH pial, RH pial, brainstem).  
   - **MNI-space** (same surfaces, plus generation of a warp field).  
2. **Color Surfaces**:  
   - **Param map projection** (nearest or linear interpolation).  
   - **Mid-thickness sampling** to color pial surfaces.  
3. **Brainstem Extraction**:  
   - Binarize specific labels in a Freesurfer aseg for T1 or MNI.  
4. **Volumetric Slicing**:  
   - Slice a 3D STL into multiple slabs along axial/coronal/sagittal orientation, for 3D printing.  
   - Optionally subdivide each slab’s mesh for denser vertex sampling (useful for coloring).  

---

## Table of Contents

- [Installation](#installation)  
- [Dependencies](#dependencies)  
- [Usage](#usage)  
  - [1. T1 Surfaces](#1-t1-surfaces)  
  - [2. MNI Surfaces](#2-mni-surfaces)  
  - [3. Brainstem Extraction](#3-brainstem-extraction)  
  - [4. Coloring Surfaces](#4-coloring-surfaces)  
  - [5. Volumetric Slicing](#5-volumetric-slicing)  
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

- `brain_for_printing_t1`
- `brain_for_printing_mni`
- `brain_for_printing_brainstem`
- `brain_for_printing_color`
- `brain_for_printing_slab_slices`

### 1) T1 Surfaces

**Generate T1-space surfaces** (LH pial, RH pial, and optionally brainstem), optionally color them with a param map:

```bash
brain_for_printing_t1 \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --output_dir /path/to/output \
  --param_map /path/to/some_T1_volume.nii.gz \
  --use_midthickness \
  --export_obj
```

- **`--subjects_dir`**: Directory containing `sub-01/anat/`  
- **`--subject_id`**: Typically the folder name inside `subjects_dir`  
- **`--no_brainstem`**: If set, skip extracting brainstem  
- **`--no_fill` / `--no_smooth`**: Skip hole-filling or Taubin smoothing on the brainstem mesh  
- **`--param_map`**: A NIfTI volume in T1 space to color the pial surfaces  
- **`--use_midthickness`**: If you have mid-thickness surfaces, color them first then copy color to pial.  

Resulting surfaces are exported as **STL** (and optionally OBJ with vertex colors).

### 2) MNI Surfaces

**Warp T1-space surfaces into MNI** and optionally color them:

```bash
brain_for_printing_mni \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --output_dir /path/to/output \
  --param_map /path/to/some_T1_volume.nii.gz \
  --use_midthickness \
  --export_obj
```

- Creates **LH pial MNI**, **RH pial MNI**, plus brainstem in MNI space.  
- Exports combined mesh as an STL or OBJ.  

### 3) Brainstem Extraction

**Extract only the brainstem** (or your custom label set) in T1 or MNI space:

```bash
brain_for_printing_brainstem \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --space T1 \
  --output_dir /path/to/output
```

You can also do `--space MNI`. By default, it uses a label set in `surfaces.py` with `--inv` logic (be sure to confirm which labels you want to keep or invert).

### 4) Coloring Surfaces

**Project a param map** onto an existing mesh (STL, OBJ, or GIFTI). For instance:

```bash
brain_for_printing_color \
  --in_mesh some_brain_mesh.stl \
  --param_map volume_in_T1.nii.gz \
  --out_obj colored_mesh.obj \
  --num_colors 6
```

This command creates a new **OBJ** with vertex colors. If the mesh is not dense enough, consider subdividing it or slicing with a smaller maximum edge length first (see “Volumetric Slicing” below).

### 5) Volumetric Slicing

**Slice a 3D mesh** into multiple slabs along a chosen orientation (axial, coronal, or sagittal). Each slab is a **3D** sub-mesh, suitable for separate 3D printing. Optionally subdivide to get more vertices for param-map coloring:

```bash
brain_for_printing_slab_slices \
  --in_mesh sub-01_T1_brain.stl \
  --orientation axial \
  --thickness 10 \
  --subdivide_max_edge 2.0 \
  --out_dir slabs_out
```

- Slicing axis: `axial` (Z), `coronal` (Y), `sagittal` (X).  
- `--thickness`: how many mm (or mesh units) per slab.  
- `--subdivide_max_edge`: if set, subdivides each slab’s triangles so no edge is longer than the given size (e.g. 2 mm). This yields a denser mesh for coloring.  
- Each slab is exported as `slab_000.stl`, `slab_001.stl`, etc.

### 6) Hollow Ventricles

Hollow out the ventricular system from a brain mesh using fMRIPrep outputs:

```bash
brain_for_printing_hollow_ventricles \
  --subjects_dir /path/to/derivatives \
  --subject_id sub-01 \
  --in_mesh sub-01_T1_brain.stl \
  --space T1 \
  --output sub-01_T1_hollowed.stl

## License

This project is licensed under the terms of the **MIT License**. 

---

## Acknowledgments

- **Neuroimaging Tools**: FSL, FreeSurfer, ANTs, MRtrix, etc.  
- **Python Libraries**: nibabel, numpy, trimesh, etc.  

---

*Thank you for using and contributing to Brain For Printing! Please [open an issue](https://github.com/claudebajada/process_brain_for_printing/issues) or reach out if you have any questions or suggestions.*

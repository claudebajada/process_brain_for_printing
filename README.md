# Brain For Printing

Generate **3 D‑printable brain surfaces** (cortical + optional brainstem) in T1 or MNI space, colour them from parametric volumes, slice them into stackable slabs, or hollow the ventricular system – all from the command line.

The core is pure‑Python (`trimesh`, `nibabel`, `scikit‑image`) while heavy lifting is delegated to familiar neuro‑tools (FreeSurfer, FSL, ANTs, MRtrix3).  Every CLI writes a timestamped JSON *run‑log* for provenance and supports structured console logging via `‑v/‑‑verbose`.

---

## Key features

| Area | Highlights |
|------|------------|
| **Surface extraction** | • LH/RH pial, white, **or** mid surfaces in T1 or warped MNI.<br>• Optional brainstem extraction with hole‑fill & smoothing.<br>• Automatic ANTs→MRtrix 4‑D warp generation. |
| **Colouring** | • Sample NIfTI param maps per‑vertex.<br>• Thresholding & discrete colour bins.<br>• Works on existing STL/OBJ/GIFTI **or** freshly generated surfaces. |
| **Mesh ops** | • Voxel remesh + repair helper.<br>• Ventricular hollowing via boolean difference.<br>• Slab slicing with per‑slab bounding‑box padding. |

---

## Installation

```bash
# 1. Clone
 git clone https://github.com/claudebajada/process_brain_for_printing.git
 cd process_brain_for_printing

# 2. Editable install + entry‑points
 pip install -e .
```

> **Python ≥3.8** recommended.  All Python dependencies are pulled in automatically.

---

## External binaries

Depending on the sub‑command you run you may need some (or all) of:

| Tool | Needed for | URL |
|------|------------|-----|
| **FreeSurfer** (`mri_convert`, `mri_binarize`) | aseg handling | <https://surfer.nmr.mgh.harvard.edu> |
| **FSL** (`fslmaths`) | mask maths | <https://fsl.fmrib.ox.ac.uk> |
| **ANTs** (`antsApplyTransforms`) | T1⇄MNI warps | <https://github.com/ANTsX/ANTs> |
| **MRtrix3** (`warpinit`, `mrcat`) | 4‑D warp field | <https://www.mrtrix.org> |
| **OpenSCAD** or **Blender** | fast boolean backend | <https://openscad.org>, <https://blender.org> |

CLIs abort early with a helpful message if a required binary is missing.

---

## Command‑line tools

| Command | Purpose |
|---------|---------|
| `brain_for_printing_cortical_surfaces` | Extract LH/RH cortical surfaces (+ brainstem) as STL. |
| `brain_for_printing_brainstem` | Brainstem‑only STL. |
| `brain_for_printing_color` | **direct** or **surface** colouring modes. |
| `brain_for_printing_slab_slices` | Slice one or more meshes into aligned slabs. |
| `brain_for_printing_hollow_ventricles` | Subtract ventricles from a brain mesh. |
| `brain_for_printing_brain_mask_surface` | Mesh a binary brain mask. |
| `brain_for_printing_combine_structures` | Merge sub‑cortical + ventricular meshes. |

Run any command with `‑h/‑‑help` for full argument reference.

---

## Quick‑start examples

### 1  Cortical surfaces + brainstem (T1)

```bash
brain_for_printing_cortical_surfaces \
  --subjects_dir /derivatives \
  --subject_id sub‑01 \
  --space T1 \
  --surf_type pial \
  --output_dir ./models \
  --split_hemis         # export LH / RH / BS separately
```

*Flags of note*  `--surf_type pial|white|mid`   `--no_brainstem`   `--no_clean`   `‑v`

### 2  Generate & colour mid‑thickness surfaces in MNI

```bash
brain_for_printing_color surface \
  --subjects_dir /derivatives \
  --subject_id sub‑01 \
  --space MNI \
  --surf_type mid \
  --param_map tstat_in_MNI.nii.gz \
  --param_threshold 2.3 \
  --output_dir ./coloured_models \
  --split_hemis \
  -v
```

### 3  Colour an existing OBJ directly

```bash
brain_for_printing_color direct \
  --in_mesh brain.obj \
  --param_map perfusion.nii.gz \
  --out_obj brain_coloured.obj
```

### 4  Slice a brain STL into 10 mm axial slabs

```bash
brain_for_printing_slab_slices \
  --in_meshes sub‑01_T1_brain.stl \
  --orientation axial \
  --thickness 10 \
  --out_dir slabs_out
```

### 5  Hollow ventricles

```bash
brain_for_printing_hollow_ventricles \
  --subjects_dir /derivatives \
  --subject_id sub‑01 \
  --in_mesh sub‑01_T1_brain.stl \
  --output sub‑01_T1_brain_hollow.stl \
  --dilate_mask \
  -v
```

---

## Logging & reproducibility

* Real‑time logging is controlled by `‑v/‑‑verbose`.
* Every run writes a JSON log (`*_log_<timestamp>.json`) capturing inputs, parameters, steps, warnings, and outputs.
* Temporary folders are auto‑removed unless you pass `--no_clean`.

---

## Roadmap

* **Unit tests** – pytest suite with tiny dummy meshes (planned).

---

## License

MIT © Claude J. Bajada and contributors

---

## Acknowledgements

* Neuroimaging software: FSL, FreeSurfer, ANTs, MRtrix3.
* Python libraries: numpy, nibabel, scipy, scikit‑image, trimesh.


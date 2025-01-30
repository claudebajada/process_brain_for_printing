# process_brain_for_printing

A Python command-line tool to prepare, merge, and optionally color 3D brain surfaces for 3D printing, in both T1 and MNI spaces. This script extracts the pial surfaces (left and right hemisphere) plus the brainstem, merges them into a single mesh, repairs holes, smooths, and can color the mesh based on a parameter map (e.g., a segmentation volume or functional data). We assume that you have processed your data using fmriprep.

---

## Overview

- **Generates T1 and/or MNI space surfaces** from preprocessed anatomical data.  
- **Merges left hemisphere, right hemisphere, and brainstem** into a single mesh for 3D printing.  
- **Optionally colors the resulting mesh** by sampling a parameter map (such as a segmentation or functional volume).  
- **Supports direct sampling on pial** or sampling on a mid-thickness surface and copying those vertex colors to pial.  
- **Exports STL** (for 3D printing) and, if requested, **OBJ** (for colored visualization).  

---

## Prerequisites

1. **Python 3.7+**  
2. **Python Packages**:  
   - [Nibabel](https://nipy.org/nibabel/)  
   - [Numpy](https://numpy.org/)  
   - [Scipy](https://scipy.org/)  
   - [scikit-image](https://scikit-image.org/)  
   - [trimesh](https://trimsh.org/)  
   - [matplotlib](https://matplotlib.org/) (for colormap functionality)  

3. **External Neuroimaging Tools** (these must be installed and available in your system `PATH`):  
   - **[Freesurfer](https://surfer.nmr.mgh.harvard.edu/)** (for `mri_convert`, `mri_binarize`)  
   - **[FSL](https://fsl.fmrib.ox.ac.uk/fsl)** (for `fslmaths`)  
   - **[ANTs](https://github.com/ANTsX/ANTs)** (for `antsApplyTransforms`)  
   - **[MRtrix3](https://www.mrtrix.org/)** (for `warpinit`, `mrcat`)  

4. **Anatomical/Surfaces Data**:  
   - Processed T1w images and surfaces in both T1 and MNI spaces (e.g., outputs from fMRIPrep or FreeSurfer-based pipelines).  
   - Correct GIFTI surfaces for left and right pial, optionally mid-thickness surfaces.  
   - A labeled NIfTI file (e.g., `*_desc-aseg_dseg.nii.gz`) for the brainstem extraction.  
   - Transform files (`*.h5`) for warping between T1 and MNI spaces.  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/brain-for-printing.git
   cd brain-for-printing
   ```
2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or "venv\Scripts\activate" on Windows
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, install the packages listed above individually.)*

4. Ensure that Freesurfer, FSL, ANTs, and MRtrix commands are in your `PATH`. For example:
   ```bash
   export FREESURFER_HOME=/path/to/freesurfer
   source $FREESURFER_HOME/SetUpFreeSurfer.sh
   export PATH=/path/to/fsl/bin:$PATH
   ...
   ```
   The exact setup depends on your environment.

---

## Usage

**Basic command**:

```bash
./process_brain_for_printing.py \
  --subjects_dir /path/to/derivatives \
  --subject_id  sub-01 \
  [OPTIONS...]
```

### Key Arguments

- `--subjects_dir`  
  Path to the top-level directory containing the subject’s `anat` folder.  
- `--subject_id`  
  Subject identifier (e.g., `sub-01`).  

- `--surfaces {both,mni,t1}`  
  Generate either T1 surfaces, MNI surfaces, or both. Default: `both`.  
- `--param_map /path/to/param_map.nii.gz`  
  A parameter volume (e.g., segmentation labels, fMRI stats) to color the surfaces.  
  - If `T1` surfaces are generated, the script assumes the map is in T1 space.  
  - If `MNI` surfaces are also requested, it copies the T1-based colors to MNI (no need to warp again).  
  - If only `MNI` surfaces are requested, it warps the map from T1 to MNI.  

- `--no_fill`  
  Skip filling small holes in the extracted brainstem mesh.  
- `--no_smooth`  
  Skip Taubin smoothing of the extracted brainstem mesh.  
- `--no_clean`  
  Retain the temporary working directory instead of deleting it at the end.  
- `--use_midthickness`  
  Sample the parameter map on the mid-thickness surface (if available), then copy that color to the pial surface.  
- `--export_obj`  
  Generate a colored OBJ file in addition to the STL.  
- `--verbose`  
  Print all external command lines and their outputs (otherwise runs quietly).

### Example Command

```bash
./process_brain_for_printing.py \
    --subjects_dir /home/user/derivatives \
    --subject_id sub-01 \
    --param_map /home/user/derivatives/sub-01/anat/sub-01_run-01_desc-aparcaseg_dseg.nii.gz \
    --surfaces both \
    --use_midthickness \
    --export_obj
```

This will:
1. Create T1-space surfaces (left pial, right pial, brainstem) and merge them into one STL.  
2. Color them using the specified `aparcaseg` volume, sampling on the mid-thickness surface.  
3. Create MNI-space surfaces and copy the T1 vertex colors to them.  
4. Export both STL (uncolored) and OBJ (colored).

---

## Outputs

- **`<subject_id>_combined_brain_LH_RH_stem_T1.stl`**  
  The merged T1-space mesh (left hemisphere, right hemisphere, brainstem).  
- **`<subject_id>_combined_brain_LH_RH_stem_T1_colored.obj`** (optional)  
  The same T1-space mesh with vertex colors.  
- **`<subject_id>_combined_brain_LH_RH_stem_MNI.stl`**  
  The merged MNI-space mesh.  
- **`<subject_id>_combined_brain_LH_RH_stem_MNI_colored.obj`** (optional)  
  The same MNI-space mesh with vertex colors.  

A hidden temporary folder `_tmp_processing_XXXXXX` is created during processing, which is removed at the end (unless `--no_clean` is specified).

---

## Tips and Notes

- **Parameter Map & Interpolation**  
  - By default, the script uses nearest-neighbor interpolation (`order=0`) when sampling the parameter map. This is ideal for discrete labels (e.g., segmentation).  
  - You can edit the code (or extend the command) to use tri-linear interpolation (`order=1`) for continuous data if needed.  

- **Filling and Smoothing**  
  - The hole-filling (`trimesh.repair.fill_holes`) and smoothing (`trimesh.smoothing.filter_taubin`) steps are specifically for the brainstem portion, which often has small holes.  
  - If you have special requirements for smoothing, you can change the parameters in the code or skip it entirely with `--no_smooth`.  

- **Warping**  
  - The script relies on `antsApplyTransforms` for NIfTI volumes and `warpinit/mrcat` from MRtrix for building the final warp field.  
  - Make sure the correct transformations (`*_from-T1w_to-MNI*` or `*_from-MNI_to-T1w*`) are present in your `anat` folder.  

- **Performance**  
  - Large meshes can be memory-intensive. If you encounter memory issues, consider reducing the resolution or simplifying surfaces before final merging.  

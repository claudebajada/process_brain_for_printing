# brain_for_printing/warp_utils.py

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

from .io_utils import run_cmd

def generate_mrtrix_style_warp(mni_file, t1_file, xfm_file, 
                               out_warp="warp.nii", tmp_dir=".", 
                               verbose=False):
    """
    Generate a 4D (X, Y, Z, 3) warp field that transforms coords
    from MNI to T1 in the format expected by MRtrix tools.
    """
    print("[INFO] Generating MRtrix-style warp field ...")

    no_warp_0 = os.path.join(tmp_dir, "no-warp-0.nii")
    no_warp_1 = os.path.join(tmp_dir, "no-warp-1.nii")
    no_warp_2 = os.path.join(tmp_dir, "no-warp-2.nii")
    warp_0    = os.path.join(tmp_dir, "warp_0.nii")
    warp_1    = os.path.join(tmp_dir, "warp_1.nii")
    warp_2    = os.path.join(tmp_dir, "warp_2.nii")

    # 1) Initialize empty warp images
    run_cmd(["warpinit", mni_file, os.path.join(tmp_dir, "no-warp-[].nii")],
            verbose=verbose)

    # 2) Apply transforms for each dimension
    for i in range(3):
        run_cmd([
            "antsApplyTransforms", "-d", "3",
            "-i", f"{tmp_dir}/no-warp-{i}.nii",
            "-r", t1_file,
            "-o", f"{tmp_dir}/warp_{i}.nii",
            "-t", xfm_file,
            "-n", "Linear"
        ], verbose=verbose)

    # 3) Concatenate into a single 4D volume
    run_cmd([
        "mrcat",
        warp_0,
        warp_1,
        warp_2,
        os.path.join(tmp_dir, out_warp)
    ], verbose=verbose)

    print(f"[INFO] Created MRtrix-style warp => {os.path.join(tmp_dir, out_warp)}")


def warp_gifti_vertices(gifti_file, warp_field_file, output_gifti_file,
                        verbose=False):
    """
    Apply a 4D warp field (shape [X, Y, Z, 3]) to each vertex of a GIFTI surface.
    """
    import nibabel as nib

    print(f"[INFO] Warping GIFTI surface: {os.path.basename(gifti_file)}")

    # Load GIFTI
    gifti_img = nib.load(gifti_file)
    vertices_darray = gifti_img.darrays[0]
    vertices = vertices_darray.data  # shape (N, 3)

    # Load warp
    warp_img = nib.load(warp_field_file)
    warp_data = warp_img.get_fdata()  # shape (X, Y, Z, 3)
    warp_affine = warp_img.affine
    inv_affine = np.linalg.inv(warp_affine)

    # Convert surface vertices to voxel coords of warp
    ones = np.ones((vertices.shape[0], 1))
    vert_hom = np.hstack([vertices, ones])  # (N, 4)
    vox_hom  = (inv_affine @ vert_hom.T).T
    vox_xyz  = vox_hom[:, :3]

    # Sample the warp volume at those voxel coords
    warped_coords = np.zeros_like(vertices)
    for dim in range(3):
        vol_dim = warp_data[..., dim]
        sample_coords = [vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]]
        warped_coords[:, dim] = map_coordinates(vol_dim, sample_coords,
                                                order=1, mode='nearest')

    # Replace vertices in the GIFTI file
    vertices_darray.data = warped_coords.astype(np.float32)
    nib.save(gifti_img, output_gifti_file)
    print(f"[INFO] Warped GIFTI => {os.path.basename(output_gifti_file)}")


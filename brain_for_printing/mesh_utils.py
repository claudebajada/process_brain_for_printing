# brain_for_printing/mesh_utils.py

import os
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure

def gifti_to_trimesh(gifti_file):
    """
    Read a GIFTI surface file and return it as a trimesh.Trimesh object.
    """
    gii = nib.load(gifti_file)
    verts = gii.darrays[0].data
    faces = gii.darrays[1].data
    return trimesh.Trimesh(vertices=verts, faces=faces)


def volume_to_gifti(nifti_file, out_gifti, level=0.5):
    """
    Convert a binary mask (NIfTI) into a GIFTI surface mesh via marching_cubes.
    """
    print(f"[INFO] Running marching_cubes on: {os.path.basename(nifti_file)}")
    nii = nib.load(nifti_file)
    vol = nii.get_fdata()
    affine = nii.affine

    verts_vox, faces, _, _ = measure.marching_cubes(
        volume=vol, 
        level=level
    )

    # Convert voxel coords -> world coords
    ones = np.ones((verts_vox.shape[0], 1))
    vert_vox_hom = np.hstack([verts_vox, ones])
    vert_xyz_hom = affine @ vert_vox_hom.T
    vert_xyz = vert_xyz_hom[:3, :].T

    # Build GIFTI
    gii = nib.gifti.GiftiImage()
    coords_da = nib.gifti.GiftiDataArray(
        data=vert_xyz.astype(np.float32),
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT32'
    )
    faces_da = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_INT32'
    )
    gii.darrays.extend([coords_da, faces_da])
    nib.save(gii, out_gifti)
    print(f"[INFO] Saved GIFTI => {os.path.basename(out_gifti)}")


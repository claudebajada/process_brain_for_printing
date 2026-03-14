# brain_for_printing/mesh_utils.py

import os
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from trimesh.remesh import subdivide_to_size
import logging # Added import

L = logging.getLogger(__name__) # Added logger instance

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
    # MODIFIED: Use logger
    L.info(f"Running marching_cubes on: {os.path.basename(nifti_file)}")
    nii = nib.load(nifti_file)
    vol = nii.get_fdata()
    affine = nii.affine

    verts_vox, faces, _, _ = measure.marching_cubes(
        volume=vol,
        level=level,
        allow_degenerate=False
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
    # MODIFIED: Use logger
    L.info(f"Saved GIFTI => {os.path.basename(out_gifti)}")

def voxel_remesh_and_repair(
    mesh: trimesh.Trimesh,
    pitch: float = 0.5,
    do_smooth: bool = True,
    smooth_iterations: int = 10
) -> trimesh.Trimesh:
    """
    Convert the mesh to a watertight volume using voxel remeshing, then
    optionally fill holes, smooth, and fix normals.

    :param mesh: A trimesh.Trimesh object (may be non-watertight).
    :param pitch: Voxel size in same units as the mesh (e.g., mm).
    :param do_smooth: If True, applies Taubin smoothing after remeshing.
    :param smooth_iterations: Number of smoothing iterations.
    :return: A new watertight trimesh.Trimesh object.
    """

    # 1) Voxelize the mesh
    vox = mesh.voxelized(pitch=pitch)
    # 2) Fill internal voids (makes the volume solid)
    vox.fill()
    # 3) Convert back to a surface mesh
    remeshed = vox.marching_cubes

    # 4) Fill small holes in the resulting surface
    trimesh.repair.fill_holes(remeshed)
    # 5) Remove unused vertices/faces
    remeshed.remove_unreferenced_vertices()
    remeshed.remove_degenerate_faces()

    if do_smooth:
        # Apply Taubin smoothing to reduce stair-stepping
        trimesh.smoothing.filter_taubin(
            remeshed,
            lamb=0.5,
            nu=-0.53,
            iterations=smooth_iterations
        )

    # 6) Fix normals
    remeshed.fix_normals()

    return remeshed

def slice_mesh_into_slabs(
    mesh: trimesh.Trimesh,
    orientation: str,
    thickness: float,
    subdivide_max_edge: float = None,
    engine: str = "auto"
) -> list[trimesh.Trimesh]:
    """
    Slice a watertight mesh into multiple slabs along a specified orientation,
    optionally subdividing each slab.

    Returns a list of trimesh.Trimesh objects, one per slab (may be empty).
    """
    if engine != "auto":
        trimesh.constants.DEFAULT_WEAK_ENGINE = engine
    # Determine orientation index
    if orientation == "axial":
        axis_index = 2
    elif orientation == "coronal":
        axis_index = 1
    else:  # sagittal
        axis_index = 0

    min_val, max_val = mesh.bounds[0][axis_index], mesh.bounds[1][axis_index]

    slab_positions = []
    current = min_val
    while current < max_val:
        top = current + thickness
        if top > max_val:
            top = max_val
        slab_positions.append((current, top))
        current += thickness

    if not slab_positions:
        return []

    all_slabs = []
    for i, (lower, upper) in enumerate(slab_positions):
        # Build the bounding box for that slab
        box_center, box_extents = _build_slab_box(mesh.bounds, orientation, lower, upper)
        box_transform = trimesh.transformations.translation_matrix(box_center)
        box_mesh = trimesh.creation.box(extents=box_extents, transform=box_transform)

        # Intersection
        try:
            slab_mesh = trimesh.boolean.intersection([mesh, box_mesh])
        except Exception:
            continue
        if not slab_mesh or slab_mesh.is_empty:
            continue

        # If it's a Scene, merge geometries
        if isinstance(slab_mesh, trimesh.Scene):
            combined = trimesh.util.concatenate(
                [g for g in slab_mesh.geometry.values() if g.is_volume]
            )
            if not combined or combined.is_empty:
                continue
            slab_mesh = combined

        # Optionally subdivide
        if subdivide_max_edge and subdivide_max_edge > 0:
            v_sub, f_sub = subdivide_to_size(
                slab_mesh.vertices,
                slab_mesh.faces,
                max_edge=subdivide_max_edge
            )
            slab_mesh = trimesh.Trimesh(vertices=v_sub, faces=f_sub)

        all_slabs.append(slab_mesh)

    return all_slabs

def _build_slab_box(bounds, orientation, lower, upper):
    """
    Helper to build a bounding box (center + extents) for the specified slab
    based on orientation and mesh bounding coordinates.
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    if orientation == "axial":
        x_size = xmax - xmin
        y_size = ymax - ymin
        z_size = upper - lower
        box_center = (
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (lower + upper) / 2.0
        )
        box_extents = (x_size, y_size, z_size)
    elif orientation == "coronal":
        x_size = xmax - xmin
        y_size = upper - lower
        z_size = zmax - zmin
        box_center = (
            (xmin + xmax) / 2.0,
            (lower + upper) / 2.0,
            (zmin + zmax) / 2.0
        )
        box_extents = (x_size, y_size, z_size)
    else:  # sagittal
        x_size = upper - lower
        y_size = ymax - ymin
        z_size = zmax - zmin
        box_center = (
            (lower + upper) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0
        )
        box_extents = (x_size, y_size, z_size)
    return box_center, box_extents

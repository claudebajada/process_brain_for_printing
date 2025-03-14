# brain_for_printing/color_utils.py

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
import trimesh

from .mesh_utils import gifti_to_trimesh

def project_param_to_surface(mesh, param_nifti_path, 
                             num_colors=6, order=0):
    """
    Sample 'param_nifti' at each vertex, discretize into 'num_colors' bins, 
    and assign color to each vertex.
    """
    print(f"[INFO] Projecting param map '{param_nifti_path}' -> mesh vertices")
    param_img   = nib.load(param_nifti_path)
    param_data  = param_img.get_fdata()
    param_aff   = param_img.affine

    # Transform each vertex into voxel coordinates
    inv_aff = np.linalg.inv(param_aff)
    verts = mesh.vertices
    ones  = np.ones((verts.shape[0], 1))
    vert_hom = np.hstack([verts, ones])
    vox_hom  = (inv_aff @ vert_hom.T).T
    vox_xyz  = vox_hom[:, :3]

    # Sample volume
    param_vals = map_coordinates(
        param_data,
        [vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]],
        order=order, mode='constant', cval=0.0
    )

    # Discretize param values => color indices
    pmin, pmax = np.min(param_vals), np.max(param_vals)
    if pmin == pmax:
        color_indices = np.zeros_like(param_vals, dtype=int)
    else:
        bins = np.linspace(pmin, pmax, num_colors + 1)
        color_indices = np.digitize(param_vals, bins) - 1
        color_indices[color_indices < 0] = 0
        color_indices[color_indices >= num_colors] = num_colors - 1

    # Build discrete color table from colormap
    cmap = matplotlib.colormaps.get_cmap('viridis').resampled(num_colors)
    color_table = (cmap(range(num_colors))[:, :4] * 255).astype(np.uint8)
    vertex_colors = color_table[color_indices]

    mesh.visual.vertex_colors = vertex_colors
    return mesh


def color_pial_from_midthickness(pial_mesh_file, mid_mesh_file, 
                                 param_nifti_path, num_colors=6, order=0):
    """
    1) Load mid surface
    2) Project param map
    3) Copy color to pial surface (1:1 vertex match)
    """
    print(f"[INFO] color_pial_from_midthickness => {pial_mesh_file}")
    mid_mesh = gifti_to_trimesh(mid_mesh_file)
    mid_colored = project_param_to_surface(
        mid_mesh,
        param_nifti_path,
        num_colors=num_colors,
        order=order
    )

    pial_mesh = gifti_to_trimesh(pial_mesh_file)
    if len(mid_colored.vertices) != len(pial_mesh.vertices):
        raise ValueError("Midthickness & pial do NOT have matching # vertices!")

    pial_mesh.visual.vertex_colors = mid_colored.visual.vertex_colors.copy()
    return pial_mesh


def copy_vertex_colors(mesh_source, mesh_target):
    """
    Copy per-vertex color from 'mesh_source' to 'mesh_target'
    (assumes same # vertices & ordering).
    """
    if len(mesh_source.vertices) != len(mesh_target.vertices):
        raise ValueError("Meshes do not have matching vertex counts!")
    mesh_target.visual.vertex_colors = mesh_source.visual.vertex_colors.copy()


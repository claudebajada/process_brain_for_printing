# brain_for_printing/color_utils.py

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
import trimesh
import logging # Added import

from .mesh_utils import gifti_to_trimesh

L = logging.getLogger(__name__) # Added logger instance

def project_param_to_surface(
    mesh: trimesh.Trimesh,
    param_nifti_path: str,
    num_colors: int = 6,
    order: int = 0,
    threshold: float = None
) -> trimesh.Trimesh:
    """
    Sample a param volume (NIfTI) at each vertex of 'mesh', then color
    the mesh using discrete bins. Optionally apply a threshold so that
    vertices below threshold get a special alpha or color.

    Args:
        mesh (trimesh.Trimesh): The geometry to color in memory.
        param_nifti_path (str): Path to a NIfTI volume file.
        num_colors (int): Number of color bins.
        order (int): Interpolation order (0=nearest, 1=linear, etc.).
        threshold (float): If set, vertices whose param < threshold will have
                           alpha=128 (or you can customize).

    Returns:
        trimesh.Trimesh: The same mesh, now with .visual.vertex_colors set.
    """
    param_img  = nib.load(param_nifti_path)
    param_data = param_img.get_fdata()
    param_aff  = param_img.affine

    # Convert mesh vertices to voxel coords
    inv_aff = np.linalg.inv(param_aff)
    verts   = mesh.vertices
    ones    = np.ones((verts.shape[0], 1), dtype=verts.dtype)
    vert_hom = np.hstack([verts, ones])
    vox_hom  = (inv_aff @ vert_hom.T).T
    vox_xyz  = vox_hom[:, :3]

    # Sample the volume at each vertex
    param_vals = map_coordinates(
        param_data,
        [vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]],
        order=order,
        mode='constant',
        cval=0.0
    )

    # Discretize into color bins
    pmin, pmax = np.min(param_vals), np.max(param_vals)
    if pmin == pmax:
        # all same param => single color
        color_indices = np.zeros_like(param_vals, dtype=int)
    else:
        bins = np.linspace(pmin, pmax, num_colors + 1)
        color_indices = np.digitize(param_vals, bins) - 1
        color_indices[color_indices < 0] = 0
        color_indices[color_indices >= num_colors] = num_colors - 1

    # Build color table from a colormap (e.g. viridis)
    cmap = matplotlib.colormaps.get_cmap('viridis').resampled(num_colors)
    color_table = (cmap(range(num_colors))[:, :4] * 255).astype(np.uint8)

    # Assign each vertex a color
    vertex_colors = color_table[color_indices]

    # If threshold is provided, handle below-threshold vertices
    if threshold is not None:
        below_mask = (param_vals < threshold)
        # e.g., half-transparency for those below threshold
        vertex_colors[below_mask, 3] = 128

    mesh.visual.vertex_colors = vertex_colors
    return mesh

def color_pial_from_midthickness(
    pial_mesh_file: str,
    mid_mesh_file: str,
    param_nifti_path: str,
    num_colors: int = 6,
    order: int = 0,
    threshold: float = None
) -> trimesh.Trimesh:
    """
    1) Load mid surface
    2) Project param map (optionally with threshold)
    3) Copy color to pial surface (1:1 vertex match)
    """
    # MODIFIED: Use logger
    L.info(f"color_pial_from_midthickness => {pial_mesh_file}")
    mid_mesh = gifti_to_trimesh(mid_mesh_file)

    # Now pass 'threshold' to project_param_to_surface if provided
    mid_colored = project_param_to_surface(
        mesh=mid_mesh,
        param_nifti_path=param_nifti_path,
        num_colors=num_colors,
        order=order,
        threshold=threshold
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


def color_mesh_with_seg_and_param(mesh, segmentation_img, seg_affine,
                                  param_img=None, param_affine=None,
                                  threshold=0.0, num_colors=6):
    """
    Colours a mesh using segmentation (5TT or aseg), optionally overridden by param map.
    """

    def get_seg_color(label):
        return {
            0: [255, 190, 120],   # Cortical GM
            1: [200, 100, 100],   # Subcortical GM
            2: [255, 255, 255],   # WM
            3: [180, 180, 255],   # CSF
            4: [255, 0, 255],     # Pathological
            42: [150, 150, 150],  # Cortical GM (aseg)
            2: [150, 150, 150],   # Cortical GM (aseg)
            41: [255, 255, 255],  # WM (aseg)
            77: [255, 255, 255],  # WM (aseg)
            10: [200, 100, 100],  # Thalamus
            11: [200, 100, 100],  # Caudate
            12: [200, 100, 100],  # Putamen
            13: [200, 100, 100],  # Pallidum
            49: [200, 100, 100],
            50: [200, 100, 100],
            51: [200, 100, 100],
            52: [200, 100, 100],
            4:  [180, 180, 255],  # CSF
        }.get(int(label), [100, 100, 100])  # fallback grey

    from scipy.ndimage import map_coordinates

    if segmentation_img.ndim == 4 and segmentation_img.shape[3] == 5:
        seg_labels = np.argmax(segmentation_img, axis=3)
    else:
        seg_labels = segmentation_img

    inv_affine = np.linalg.inv(seg_affine)
    ones = np.ones((mesh.vertices.shape[0], 1))
    vert_hom = np.hstack([mesh.vertices, ones])
    vox_coords = (inv_affine @ vert_hom.T).T[:, :3].T
    sampled_labels = map_coordinates(seg_labels, vox_coords, order=0, mode='nearest')

    base_colors = np.array([get_seg_color(lbl) for lbl in sampled_labels], dtype=np.uint8)

    if param_img is not None and param_affine is not None:
        inv_param_aff = np.linalg.inv(param_affine)
        vox_param = (inv_param_aff @ vert_hom.T).T[:, :3].T
        param_vals = map_coordinates(param_img, vox_param, order=1, mode='nearest')
        mask = param_vals > threshold

        if np.any(mask):
            cmap = matplotlib.colormaps.get_cmap('viridis').resampled(num_colors)
            pmin, pmax = np.min(param_vals[mask]), np.max(param_vals[mask])
            bins = np.linspace(pmin, pmax, num_colors + 1)

            for i, val in enumerate(param_vals):
                if val > threshold:
                    bin_idx = np.digitize(val, bins) - 1
                    bin_idx = np.clip(bin_idx, 0, num_colors - 1)
                    rgb = (np.array(cmap(bin_idx)[:3]) * 255).astype(np.uint8)
                    base_colors[i] = rgb

    mesh.visual.vertex_colors = base_colors
    return mesh

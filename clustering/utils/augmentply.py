import os
import sys
import argparse  # Import argparse for command-line parsing
import nibabel as nib  # Import nibabel for handling neuroimaging files
import numpy as np  # Import numpy for numerical operations
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.interpolate import interpn

def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    """
    Resample a 3D polygon (series of points) to have a specific number of points distributed evenly along its length.

    Parameters:
    - xy (np.ndarray): The input array of points in the polygon.
    - n_points (int): The desired number of points in the resampled polygon.

    Returns:
    - np.ndarray: The array of resampled points.
    """
    # Calculate the cumulative Euclidean distance between successive points
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])
    # Linearly spaced distances along the total distance
    d_sampled = np.linspace(0, d.max(), n_points)
    # Interpolate x, y, z coordinates along the distances
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
        np.interp(d_sampled, d, xy[:, 2]),
    ]
    return xy_interp


def find_subject_descriptor(path):
    """
    Function to find the sub-XXXXX descriptor and convert it to float32
    """
    for part in path.parts:
        if part.startswith("sub-") and part[4:9].isdigit():
            return int(part[4:9])
    return None  # Return None if no matching part is found

# TODO: Verify, this is correct
def interpolate_data(nifti, streamlines):
    """Interpolate the data at the given points."""
    data = nib.load(nifti).get_fdata()
    affine = nib.load(nifti).affine
    inv_affine = np.linalg.inv(affine)
    vertices = np.vstack(streamlines)
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    coords = np.dot(inv_affine, vertices.T).T[:, :3]
    interp = interpn((np.arange(data.shape[0]), np.arange(data.shape[1]), np.arange(data.shape[2])), data, coords, method='linear', bounds_error=False, fill_value=0)
    return interp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inply', type=str, help='input .ply file for processing')
    parser.add_argument('-o', '--outply', type=str, help='output path for the .ply file')
    parser.add_argument('--fa', type=str, help='path to the fractional anisotropy.')
    parser.add_argument('--ad', type=str, help='path to the axial diffusivity.')
    parser.add_argument('--rd', type=str, help='path to the radial diffusivity.')
    parser.add_argument('--md', type=str, help='path to the mean apparent diffusion file.')
    parser.add_argument('--l1', type=str, help='path to the fractional anisotropy.')
    parser.add_argument('--l2', type=str, help='path to the axial diffusivity.')
    parser.add_argument('--l3', type=str, help='path to the radial diffusivity.')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    parser.add_argument('-n', '--n_supports', default=100, help='Number of points to which each streamline is resampled')
    
    args = parser.parse_args()  # Parse command line arguments
    n_supports = args.n_supports
    outply = Path(args.outply)  # Set the output filename from argument
    inply = Path(args.inply)
    feature_paths = {
        'fa' : args.fa, 
        'ad' : args.ad, 
        'rd' : args.rd, 
        'md' : args.md,
        'l1' : args.l1, 
        'l2' : args.l2, 
        'l3' : args.l3, 
    }
   
    assert inply.suffix == '.ply', '.ply, with vertex subject labeling expected'

    # Read streamline data 
    plydata = PlyData.read(inply)
    vertices = np.vstack([plydata['vertices']['x'], plydata['vertices']['y'], plydata['vertices']['z']]).T
    end_indices = plydata['fiber']['endindex']
    streamlines = [vertices[start:end] for start, end in zip([0] + end_indices[:-1].tolist(), end_indices)] 
    streamlines = np.array([resample_polygon(s, n_supports) for s in streamlines])  # Resample each streamline    
    n_streamlines = len(streamlines) 

   # Interpolate on vertices
    vertex_features = {label: interpolate_data(file_path, streamlines) for label, file_path in feature_paths.items()}

    # Streamline level features
    end_indices = np.cumsum(np.array([len(s) for s in streamlines]))
    sub_ids = [find_subject_descriptor(inply) for i in range(len(end_indices))]
    side_ids = np.zeros(len(sub_ids)) if 'lh' in inply.parts[-1] else np.ones(len(sub_ids)) # TODO: check whether this works correctly


    vertices = np.vstack(streamlines)

    # Set datatypes, assign data, and write to .ply file
    vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + [(feature, 'f4') for feature in vertex_features.keys()]
    fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4'), ('sideid', 'i4')]

    ply_vertices = np.empty(vertices.shape[0], dtype=vertex_dtypes)
    ply_vertices['x'] = vertices[:, 0]
    ply_vertices['y'] = vertices[:, 1]
    ply_vertices['z'] = vertices[:, 2]
    for feature in vertex_features.keys():
        ply_vertices[feature] = vertex_features[feature]

    ply_fibers = np.empty(n_streamlines, dtype=fiber_dtypes)
    ply_fibers['endindex'] = end_indices
    ply_fibers['subid'] = sub_ids
    ply_fibers['sideid'] = side_ids
    
    vertices = PlyElement.describe(ply_vertices, 'vertices') 
    fibers = PlyElement.describe(ply_fibers, 'fiber')

    PlyData([vertices, fibers], text=False).write(outply)
    #print(f'Wrote .ply file with features to {outply}')
    #print(f'Wrote .ply: {os.path.getsize(outply)/1000000} Mb')
    

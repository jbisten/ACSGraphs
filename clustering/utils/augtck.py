import os
import sys
import argparse  # Import argparse for command-line parsing
import nibabel as nib  # Import nibabel for handling neuroimaging files
import numpy as np  # Import numpy for numerical operations
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.interpolate import interpn
import h5py
from bonndit.utils.tck_io import Tck
import re

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
    parser.add_argument('-i', '--infile', type=str, help='input .tck file for processing')
    parser.add_argument('-o', '--outfile', type=str, help='output path for the .h5 file')
    parser.add_argument('--fa', type=str, help='path to the fractional anisotropy.')
    parser.add_argument('--ad', type=str, help='path to the axial diffusivity.')
    parser.add_argument('--rd', type=str, help='path to the radial diffusivity.')
    parser.add_argument('--md', type=str, help='path to the mean apparent diffusion file.')
    parser.add_argument('--l2', type=str, help='path to the axial diffusivity.')
    parser.add_argument('--l3', type=str, help='path to the radial diffusivity.')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    parser.add_argument('-n', '--n_supports', default=50, help='Number of points to which each streamline is resampled')
    
    args = parser.parse_args()  # Parse command line arguments
    n_supports = args.n_supports
    outfile = Path(args.outfile)  # Set the output filename from argument
    infile = Path(args.infile)
    feature_paths = {
        'fa' : args.fa, 
        'ad' : args.ad, 
        'rd' : args.rd, 
        'md' : args.md,
        'l2' : args.l2, 
        'l3' : args.l3, 
    }
   
    assert infile.suffix == '.tck', '.tck file is expected'

     # Read streamline data
    tck = nib.streamlines.load(infile)
    streamlines = tck.streamlines

    # Resample each streamline
    streamlines = np.array([resample_polygon(s, n_supports) for s in streamlines])

    # Write .tck-file
    tck = Tck(str(infile))
    tck.force = True
    tck.write({})
    [tck.append(s, None) for s in streamlines]
    tck.close()

    n_streamlines = len(streamlines)

    # Interpolate on vertices
    vertex_features = {label: interpolate_data(file_path, streamlines) for label, file_path in feature_paths.items()}

    # Streamline level features
    extract_sub_id = lambda f: re.search(r'sub-000([0-4][0-9]|50)', f).group(0) 
    sub_id = extract_sub_id(str(infile))
    side_id = 'lh' if 'lh' in infile.parts[-1] else 'rh'

    # Create HDF5 file
    with h5py.File(outfile, 'w') as f:
        # Create datasets
        for feature in vertex_features.keys():
            f.create_dataset(feature, data=vertex_features[feature])

        # Optionally, add some metadata
        f.attrs['n_streamlines'] = n_streamlines
        f.attrs['n_supports'] = args.n_supports
        f.attrs['sub_id'] = sub_id 
        f.attrs['side_id'] = side_id 
        for feature in feature_paths.keys():
            f.attrs[f'mean_{feature}'] = vertex_features[feature].mean() 

    print(f"HDF5 file created with {n_streamlines} streamlines. Streamlines resampled to {n_supports} points per streamline.")
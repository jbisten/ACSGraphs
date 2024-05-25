import os
import sys
import argparse  # Import argparse for command-line parsing
import nibabel as nib  # Import nibabel for handling neuroimaging files
import numpy as np  # Import numpy for numerical operations
from pathlib import Path
from plyfile import PlyData, PlyElement

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infiles', type=str, action='append', help='input .ply file for processing')
    parser.add_argument('-o', '--outfile', type=str, help='output path for the .ply file')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    parser.add_argument('-n', '--n_supports', default=100, help='Number of points to which each streamline is resampled')
    
    args = parser.parse_args()  # Parse command line arguments
    outfile = Path(args.outfile)  # Set the output filename from argument
    n_supports = args.n_supports

    vertices = []
    sub_ids = []
    end_indices = []
    vertex_features = {}

    end_idx_offset = 0
    n_streamlines = 0    
    for ply_in in args.infiles:
        ply_in = Path(ply_in)
        assert ply_in.suffix == '.ply', "file is not a .ply-file"

        # Read file
        plydata = PlyData.read(ply_in)
        # Add new vertices
        vertices.append(np.vstack([plydata['vertices'][field] for field in ['x', 'y', 'z']]).T)
        
        # Add new end indices with the addition of the offset
        end_indices.append(plydata['fiber']['endindex'] + end_idx_offset)
        n_streamlines += len(plydata['fiber']['endindex'])
        end_idx_offset += len(plydata['vertices']['x'])
       
        # Add subids
        sub_ids.append(plydata['fiber']['subid'])
        
        # Extract and add vertex data excluding x, y, z
        ply_vertex_features = plydata['vertices'].data.dtype.names
        for feature in ply_vertex_features:
            if feature not in ['x', 'y', 'z']:
                if feature not in vertex_features:
                    vertex_features[feature] = []
                vertex_features[feature].append(plydata['vertices'][feature])

    # Set datatypes, assign data, and write to .ply file
    vertices = np.vstack(vertices)
    end_indices = np.array(end_indices)
    sub_ids = np.array(sub_ids)
    side_ids = np.zeros(len(sub_ids)) if 'lh' in inply.parts[-1] else np.ones(len(sub_ids))
    # Concatenate all vertex features
    for feature in vertex_features.keys():
        vertex_features[feature] = np.concatenate(vertex_features[feature])


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

    PlyData([vertices, fibers], text=False).write(outfile)
    #print(f'Concatenated mni-warped .ply file with features to {outfile}')
    #print(f'Wrote {os.path.getsize(outfile)/1000000} Mb')

    

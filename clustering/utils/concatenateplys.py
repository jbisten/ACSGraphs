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


def find_subject_descriptor(path):
    """
    Function to find the sub-XXXXX descriptor and convert it to float32
    """
    for part in path.parts:
        if part.startswith("sub-") and part[4:9].isdigit():
            return int(part[4:9])
    return None  # Return None if no matching part is found

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infiles', type=str, action='append', help='input .ply file for processing')
    parser.add_argument('-o', '--outfile', type=str, help='output path for the .ply file')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    parser.add_argument('-n', '--n_supports', default=21, help='Number of points to which each streamline is resampled')
    
    args = parser.parse_args()  # Parse command line arguments
    outfile = args.outfile  # Set the output filename from argument
    n_supports = args.n_supports

    streamlines = []
    sub_ids = []
    end_indices = []

    for tck_in in args.infiles:
        print(tck_in)
        tck_path = Path(tck_in)
        if nib.streamlines.detect_format(tck_in) is not nib.streamlines.TckFile:
            print("Skipping non-TCK file: '{}'".format(tck_in))
            continue

        if os.path.isfile(outfile) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(outfile))
            break

        tck = nib.streamlines.load(tck_path) # Load the .tck file using nibabel
        
        # Iterate over tck and
        end_idx = 0 
        for x in tck.streamlines:
            assert find_subject_descriptor(tck_path) != None, 'No subject descriptor found, sub label is None!'
            end_idx += n_supports
            streamlines.append(resample_polygon(x, n_supports))  # Resample each streamline
            sub_ids.append(find_subject_descriptor(tck_path)) # Streamline level annotation 
            #[sub_ids.append(find_subject_descriptor(tck_path)) for n  in range(n_supports)] # Vertex level annotation

    streamlines = np.array(streamlines, dtype=np.float32)  # Convert the list to a numpy array
    end_indices = np.cumsum(np.array([len(s) for s in streamlines]))
    streamlines = np.vstack(streamlines)

    fiber_meta = np.array([end_indices, sub_ids]).T

    vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4')]

    ply_vertices = np.empty(streamlines.shape[0], dtype=vertex_dtypes)
    ply_vertices['x'] = streamlines[:, 0]
    ply_vertices['y'] = streamlines[:, 1]
    ply_vertices['z'] = streamlines[:, 2]

    ply_fibers = np.empty(fiber_meta.shape[0], dtype=fiber_dtypes)
    ply_fibers['endindex'] = fiber_meta[:, 0]
    ply_fibers['subid'] = fiber_meta[:, 1]

    vertices = PlyElement.describe(ply_vertices, 'vertices') 
    fibers = PlyElement.describe(ply_fibers, 'fiber')

    PlyData([vertices, fibers], text=False).write(outfile)
    print(f'Wrote {len(args.infiles)} ply files to concatenated .ply file at {outfile}')
    print(f'Wrote .ply: {os.path.getsize(outfile)/1000000} Mb')
    

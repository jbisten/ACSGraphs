import os
import sys
import argparse  # Import argparse for command-line parsing
import nibabel as nib  # Import nibabel for handling neuroimaging files
sys.path.append(os.path.abspath('..'))
import IOFibers as io  # Import a custom module for fiber bundle operations
import numpy as np  # Import numpy for numerical operations
from pathlib import Path

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

# Function to find the sub-XXXXX descriptor and convert it to float32
# Function to find the sub-XXXXX descriptor
def find_subject_descriptor(path):
    for part in path.parts:
        if part.startswith("sub-") and part[4:].isdigit():
            return part
    return None  # Return None if no matching part is found

def main():
    """
    Main function to execute the script logic based on command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infiles', type=str, action='append', help='input .tck file for processing')
    parser.add_argument('-o', '--outfile', type=str, help='output path for the .bundle file')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')

    args = parser.parse_args()  # Parse command line arguments
    outfile = args.outfile  # Set the output filename from argument

    all_bundles = []
    bundle_ids = []
    for tck_in in args.infiles:
        tck_path = Path(tck_in)
        if nib.streamlines.detect_format(tck_in) is not nib.streamlines.TckFile:
            print("Skipping non-TCK file: '{}'".format(tck_in))
            continue

        if os.path.isfile(outfile) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(outfile))
            return

        tck = nib.streamlines.load(tck_path) # Load the .tck file using nibabel
        
        # Iterate over tck and  
        for x in tck.streamlines:
            all_bundles.append(resample_polygon(x, 21))  # Resample each streamline
            assert find_subject_descriptor(tck_path) != None, 'No subject descriptor found, sub label is None!'
            bundle_ids.append(find_subject_descriptor(tck_path))
            

    all_bundles = np.array(all_bundles, dtype=np.float32)  # Convert the list to a numpy array
    all_bundles = np.expand_dims(all_bundles, 0)  # Add a new axis to the array

    io.write_bundles(outfile, all_bundles, bundle_ids)  # Write the result to a .bundle file

    print(f'Wrote {len(args.infiles)} tck files to .bundles file at {outfile}')
    print(f'Wrote {os.path.getsize(outfile)} Bytes')

if __name__ == '__main__':
    main()

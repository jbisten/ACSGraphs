import os
import sys
import argparse  # Import argparse for command-line parsing
import nibabel as nib  # Import nibabel for handling neuroimaging files
sys.path.append(os.path.abspath('..'))
import IOFibers as io  # Import a custom module for fiber bundle operations
import numpy as np  # Import numpy for numerical operations


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

def build_argparser():
    """
    Build and return an argparse.ArgumentParser for command-line arguments parsing.

    Returns:
    - argparse.ArgumentParser
    """
    DESCRIPTION = "Convert tractograms (.tck -> .bundle)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-i', '--input', type=str, help='input .tck file for processing')
    p.add_argument('-o', '--output', type=str, help='output path for the .bundle file')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p

def main():
    """
    Main function to execute the script logic based on command line arguments.
    """
    parser = build_argparser()  # Build the argument parser
    args = parser.parse_args()  # Parse command line arguments

    if nib.streamlines.detect_format(args.input) is not nib.streamlines.TckFile:
        print("Skipping non-TCK file: '{}'".format(args.input))
        return

    output_filename = args.output  # Set the output filename from argument
    if os.path.isfile(output_filename) and not args.force:
        print("Skipping existing file: '{}'. Use -f to overwrite.".format(output_filename))
        return

    tck = nib.streamlines.load(args.input)  # Load the .tck file using nibabel
    my_bundle = []
    for x in tck.streamlines:
        my_bundle.append(resample_polygon(x, 21))  # Resample each streamline
    my_bundle = np.array(my_bundle, dtype=np.float32)  # Convert the list to a numpy array
    my_bundle = np.expand_dims(my_bundle, 0)  # Add a new axis to the array
    print(my_bundle.shape)  # Print the shape of the result
    io.write_bundles(output_filename, my_bundle)  # Write the result to a .bundle file

if __name__ == '__main__':
    main()

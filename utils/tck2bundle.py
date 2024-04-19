import os
import sys
sys.path.append('/home/xderes/')
import argparse
import nibabel as nib
import FFClust.IOFibers as io

import numpy as np


def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), n_points)

    # interpolate x and y coordinates
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
        np.interp(d_sampled, d, xy[:, 2]),
    ]

    return xy_interp

def build_argparser():
    DESCRIPTION = "Convert tractograms (TCK -> Bundle)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('tractograms', metavar='bundle', nargs="+", help='list of tractograms.')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    for tractogram in args.tractograms:
        if nib.streamlines.detect_format(tractogram) is not nib.streamlines.TckFile:
            print("Skipping non TRK file: '{}'".format(tractogram))
            continue

        output_filename = tractogram[:-4] + '.bundle'
        if os.path.isfile(output_filename) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(output_filename))
            continue

        tck = nib.streamlines.load(tractogram)
        my_bundle=[]
        for x in tck.streamlines:
            my_bundle.append(resample_polygon(x,21))
        my_bundle = np.array(my_bundle, dtype=np.float32)
        my_bundle = np.expand_dims(my_bundle,0)
        print(my_bundle.shape)
        io.write_bundles(output_filename, my_bundle)

if __name__ == '__main__':
    main()

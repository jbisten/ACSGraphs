import os
import sys
sys.path.append(os.path.abspath('..'))
import argparse
import nibabel as nib
import IOFibers as io
from tqdm import tqdm
import numpy as np
from bonndit.utils.tck_io import Tck 


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('tractograms', metavar='bundle', help='list of tractograms.')
    parser.add_argument('-i', '--infile')
    parser.add_argument('-o', '--outfile')
    args = parser.parse_args()

    ##############
    # Read bundles
    ##############
    bundles, _, _ = io.read_bundles(args.infile)
    bundles = bundles.squeeze(1)

    ###########
    # Write Tck
    ###########
    tck = Tck(args.outfile)
    tck.force = True
    tck.write({})
    [tck.append(bundles[i,:, :], None) for i in range(len(bundles))]
    tck.close()
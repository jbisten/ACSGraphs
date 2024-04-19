import os
import sys
sys.path.append('/home/xderes/')
import argparse
import nibabel as nib
import FFClust.IOFibers as io
from tqdm import tqdm
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
    p.add_argument('tractograms', metavar='bundle', help='list of tractograms.')
    p.add_argument('-i')
    p.add_argument('-o')
    p.add_argument('-ts')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    bundles, _ = io.read_bundles(args.i)
    bundles_ref, _ = io.read_bundles(args.ts)
    bundles_ref = bundles_ref[0]
    tck = nib.streamlines.load(args.tractograms)
    tck = tck.streamlines
    os.mkdir(args.o)
    for i, x in tqdm(enumerate(bundles), total=len(bundles)):
        my_tracto= [np.where(np.linalg.norm(np.linalg.norm(bundles_ref-y, axis=1), axis=1) < 0.01)[0][0] for y in x]
        streamlines = nib.streamlines.ArraySequence(tck[my_tracto])

        mytractogram = nib.streamlines.tractogram.Tractogram(streamlines, affine_to_rasmm=np.identity(4))
        tractogram = nib.streamlines.tck.TckFile(mytractogram)
        tractogram.save(args.o + '/' + str(i) + '.tck')
        tck = np.delete(tck, my_tracto, 0)
        bundles_ref = np.delete(bundles_ref, my_tracto, 0)


if __name__ == '__main__':
    main()


import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path
import nibabel as nib
from bonndit.utils.tck_io import Tck
from nibabel.streamlines import ArraySequence, save
from dipy.segment.clustering import QuickBundles
from plyfile import PlyData, PlyElement
import os
from tqdm import tqdm
import sys 
sys.path.append(os.path.abspath("/home/justus/codebase/ACSGraphs/clustering/utils/"))
from clusterIO import load_tck, load_hdf5, save_cluster_representatives


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform clustering on a dataset of streamlines')
    parser.add_argument('--infiles', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('--inhdf5s', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('--k', type=int, default=7, help='Number of clusters to be used for each point in K-Means for map')
    parser.add_argument('--outdir', help='Directory where to place all the results')
    parser.add_argument('--tsne_analysis', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Arguments
    infiles = [Path(f) for f in args.infiles]
    inhdf5s = [Path(f) for f in args.inhdf5s]
    outdir = Path(args.outdir)
    k = args.k

    # TODO: Choose appropriate array type to collect streamlines
    streamlines = []
    h5_data = []

    # Loading all streamlines into memory
    for infile in tqdm(infiles, total=len(infiles), desc='Loading .tck files'): 
        assert infile.suffix == '.tck', '.tck, with vertex subject labeling expected'

        # Append streamlines to streamline array for Clustering
        streamlines.extend(load_tck(infile)[1])

    #################
    # Quickclustering
    #################
    qb = QuickBundles(threshold=22.)
    clusters = qb.cluster(streamlines)

    print(f"Found {len(clusters)} clusters")
    cluster_labels=np.zeros(len(streamlines))
    for n,c in enumerate(clusters):
       for i in c.indices:
           cluster_labels[i] = n


    # Write output .tck files
    save_cluster_representatives(np.array(streamlines), k=len(clusters), cluster_labels=cluster_labels, outdir=outdir)           

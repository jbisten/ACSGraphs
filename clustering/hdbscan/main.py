import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path
import nibabel as nib
from nibabel.streamlines import ArraySequence, save
from plyfile import PlyData, PlyElement
import os 
import cuml
os.sys.path.append(os.path.abspath("/home/justus/codebase/ACSGraphs/clustering/utils/"))
from clusterIO import load_hdf5, save_cluster_representatives, load_tck 
import torch
from tqdm import tqdm

def get_min_dist_streams(streamlines):
    # Convert numpy ndarray to PyTorch tensor if it's not already a tensor
    if not isinstance(streamlines, torch.Tensor):
        streamlines = torch.tensor(streamlines, dtype=torch.float32)

    # Select the first streamline as the reference
    ref_streamline = streamlines[0]

    # Compute distances to the non-flipped version of the reference
    dist_non_flipped = torch.norm(streamlines - ref_streamline, dim=2).sum(dim=1)

    # Compute distances to the flipped version of the reference
    ref_streamline_flipped = torch.flip(ref_streamline, dims=[0])
    dist_flipped = torch.norm(streamlines - ref_streamline_flipped, dim=2).sum(dim=1)

    # Determine the minimal distances and select accordingly
    min_dist, indices = torch.min(torch.stack([dist_non_flipped, dist_flipped], dim=1), dim=1)

    # Use these indices to flip streamlines where the flipped version was closer
    min_dist_streamlines = torch.where(indices.unsqueeze(1).unsqueeze(2).expand_as(streamlines) == 1,
                                       torch.flip(streamlines, dims=[1]),
                                       streamlines)
    
    return min_dist_streamlines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform clustering on a dataset of streamlines')
    parser.add_argument('--infiles', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('--inhdf5s', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('--outdir', help='Directory where to place all the results')
    parser.add_argument('--tsne_analysis', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Arguments
    infiles = [Path(f) for f in args.infiles]
    inhdf5s = [Path(f) for f in args.inhdf5s]
    outdir = Path(args.outdir)

    # TODO: Choose appropriate array type to collect streamlines
    streamlines = []
    h5_data = []

    # Loading all streamlines into memory
    for infile in tqdm(infiles, total=len(infiles), desc='Loading .tck files'): 
        assert infile.suffix == '.tck', '.tck, with vertex subject labeling expected'

        # Append streamlines to streamline array for Clustering
        streamlines.extend(load_tck(infile)[1])

    print('Streamlines loaded, minimizing flip distances . . .')

    # Minimize distances between streamlines
    streamlines = torch.Tensor(get_min_dist_streams(np.array(streamlines))).to('cuda:0') 
    streamlines = streamlines.reshape(streamlines.shape[0], -1)

    print(streamlines.shape)
    #################
    #    HDBSCAN
    #################
    hdb = cuml.cluster.hdbscan.HDBSCAN(cluster_selection_epsilon=100, cluster_selection_method='leaf')
    hdb.fit_predict(streamlines)


    print('Number of clusters: ', max(hdb.labels_))
    print(sorted([(element, count) for element, count in zip(*np.unique(hdb.labels_, return_counts=True))], key=lambda x: x[1], reverse=True)[:5])

    # TODO: Write tck with representative cluster streamlines
    save_cluster_representatives(streamlines, k=max(hdb.labels_), cluster_labels=hdb.labels_, outdir=outdir)
       
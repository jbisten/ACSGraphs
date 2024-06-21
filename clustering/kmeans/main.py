from clustering import gpu_sc, get_min_dist_streams, get_cluster_centroids
from pathlib import Path
import argparse
from plyfile import PlyData, PlyElement
import nibabel as nib
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("/home/justus/codebase/ACSGraphs/clustering/utils/"))
from clusterIO import load_tck, load_hdf5, save_cluster_representatives
from tqdm import tqdm
import h5py

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

    print('Streamlines loaded, minimizing flip distances . . .')

    # Minimize distances between streamlines
    streamlines = get_min_dist_streams(np.array(streamlines))   

    if args.tsne_analysis:
        feature_vectors = np.empty()
        for i in range(1000):
            cluster_ids, _ = gpu_sc(streamlines, k)
        # TODO: Write to hdf5 


    # Clustering
    elif not args.tsne_analysis:
        # Clustering and computation of Cluster Centers
        cluster_ids, cluster_centers = gpu_sc(streamlines, k)
        cluster_centroids = get_cluster_centroids(streamlines, k, cluster_ids)

        offset = 0
        for hdf5 in tqdm(inhdf5s, total=len(inhdf5s), desc='Adding cluster labels to .h5-files'):
            hdf5_data = load_hdf5(hdf5)
            n_streamlines=hdf5_data['n_streamlines']
            cluster_labels=cluster_ids[offset:offset+n_streamlines].cpu().numpy()
            offset+=n_streamlines

            # Create Updated HDF5 file with cluster labels for each streamline
            with h5py.File(hdf5, 'a') as f:
                if 'cluster_labels' in f:
                    del f['cluster_labels']
                # Add the new dataset for cluster labels
                f.create_dataset('cluster_labels', data=cluster_labels)
        
        
        # TODO: Write tck with representative cluster streamlines
        save_cluster_representatives(streamlines, k=k, cluster_labels=cluster_labels, outdir=outdir)
       
        # TODO: Write tck with cluster centroids        
        # save_centroid_tck(cluster_centroids, filename=f'{str(outdir / "kmeans_centroids.ply")}')

        # TODO: Write tck with representative cluster streamlines        
        
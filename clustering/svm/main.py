
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
import neuroutils


def clustersvm(streamlines, nu):
    length_array=[]
    filtered_streamlines=[]
    for s in streamlines:
         if len(s) > 10:
             length_array.append(len(s))
             filtered_streamlines.append(s)
    all_data = np.vstack(filtered_streamlines)
    all_idx = np.cumsum(length_array)

    stream_features = neuroutils.covMatrixOptimized(all_data, all_idx)
    
    mean = np.mean(stream_features, axis=0)

    std = np.std(stream_features, axis=0)

    stream_features = np.divide(stream_features - mean, std, where=std!=0)

    Fnorm, numFiber, mask = neuroutils.GPUoutlierSVM(stream_features, nu)
 
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform clustering on a dataset of streamlines')
    parser.add_argument('--infiles', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('--inhdf5s', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('--k', type=int, default=3, help='Number of clusters to be used for each point in K-Means for map')
    parser.add_argument('--outdir', help='Directory where to place all the results')
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

    streamlines = np.array(streamlines)

    print('Streamlines loaded, minimizing flip distances . . .')

    # Clustering
    n = int(streamlines.shape[0] / k) # n streamlines per cluster
    cstreams = streamlines # Streamlines to be clustered
        
    original_indices = np.arange(streamlines.shape[0])
    cluster_labels = np.zeros(streamlines.shape[0], dtype=int)

    for i in range(1, k):
        # TODO: Choose nu to classify n streamlines
        nu = n / cstreams.shape[0]
        print(f'{i}: Running One-Class SVM with nu {nu}')

        # TODO: Run one-class SVM 
        inliers_mask = clustersvm(cstreams, nu) > 0
        cluster_mask = ~inliers_mask
            
        # Update the outlier_iteration array with the current iteration for new outliers
        cluster_labels[cluster_mask] = i

        # TODO: Find a way to mark the original positions of my streamlines in the array!
        cstreams = cstreams[cluster_mask]
        inliers = cstreams[inliers_mask]
            
        # Keep only the inliers for the next iteration
        original_indices = original_indices[inliers_mask] 

    unique_elements, counts = np.unique(cluster_labels, return_counts=True)
        
    for element, count in zip(unique_elements, counts):
        print(f"Element: {element}, Count: {count}")

    # Clustering and computation of Cluster Centers
    #cluster_ids, cluster_centers = gpu_sc(streamlines, k)
    #cluster_centroids = get_cluster_centroids(streamlines, k, cluster_ids)

    #offset = 0
    #for hdf5 in tqdm(inhdf5s, total=len(inhdf5s), desc='Adding cluster labels to .h5-files'):
    #    hdf5_data = load_hdf5(hdf5)
    #    n_streamlines=hdf5_data['n_streamlines']
    #    cluster_labels=cluster_ids[offset:offset+n_streamlines].cpu().numpy()
    #    offset+=n_streamlines

    #    # Create Updated HDF5 file with cluster labels for each streamline
    #    with h5py.File(hdf5, 'a') as f:
    #        if 'cluster_labels' in f:
    #            del f['cluster_labels']
    #        # Add the new dataset for cluster labels
    #        f.create_dataset('cluster_labels', data=cluster_labels)
        
        
    # TODO: Write tck with representative cluster streamlines
    #save_cluster_representatives(streamlines, k=k, cluster_labels=cluster_labels, outdir=outdir)
       
    # TODO: Write tck with cluster centroids        
    # save_centroid_tck(cluster_centroids, filename=f'{str(outdir / "kmeans_centroids.ply")}')

    # TODO: Write tck with representative cluster streamlines        
        
from clustering import gpu_sc, get_min_dist_streams, get_cluster_centroids
from pathlib import Path
import argparse
from plyfile import PlyData, PlyElement
import nibabel as nib
import numpy as np
from bonndit.utils.tck_io import Tck
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from plyio import save_centroid_ply, save_cluster_ply

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform clustering on a dataset of streamlines')
    parser.add_argument('--infile', help='Input streamlines file')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters to be used for each point in K-Means for map')
    parser.add_argument('--outdir', help='Directory where to place all output')
    args = parser.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    k = args.k

    assert infile.suffix == '.ply', '.ply, with vertex subject labeling expected'
    
    plydata = PlyData.read(infile)
    vertices = np.vstack([plydata['vertices']['x'], plydata['vertices']['y'], plydata['vertices']['z']]).T
    end_indices = plydata['fiber']['endindex']
    sub_ids = plydata['fiber']['subid']
    streamlines= np.array([vertices[i - 21:i] for i in end_indices]) 

    # Minimize distances between streamlines
    streamlines = get_min_dist_streams(streamlines) 

    # Clustering and computation of Cluster Centers
    cluster_ids, cluster_centers = gpu_sc(streamlines, k)
    cluster_centroids = get_cluster_centroids(streamlines, k, cluster_ids)

    # Write output .ply files
    save_centroid_ply(cluster_centroids, filename=f'{str(outdir / "kmeans_centroids.ply")}')
    save_cluster_ply(streamlines, sub_ids, cluster_ids, filename=f'{str(outdir / "kmeans_clusters.ply")}')

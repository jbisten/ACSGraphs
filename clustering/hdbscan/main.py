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
from plyio import save_centroid_ply, save_cluster_ply, save_individual_clusters, load_ply_file 

if __name__ == "__main__":
    # Argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Input file in .ply format")
    parser.add_argument("-o", "--outdir", help="Output directory")
    parser.add_argument("-n", "--n_supports", default=100, help="Output directory")
    
    args = parser.parse_args()
    outdir = Path(args.outdir)
    infile = Path(args.infile)
    n_supports = args.n_supports

    vertices, streamlines, end_ids, sub_ids, side_ids, vertex_features = load_ply_file(args.infile) 

    streamlines = streamlines.reshape(streamlines.shape[0], -1)

    print(streamlines.shape)
    #################
    #    HDBSCAN
    #################
    hdb = cuml.cluster.hdbscan.HDBSCAN(cluster_selection_epsilon=100, cluster_selection_method='leaf')
    hdb.fit_predict(streamlines)


    print('Number of clusters: ', max(hdb.labels_))
    print(sorted([(element, count) for element, count in zip(*np.unique(hdb.labels_, return_counts=True))], key=lambda x: x[1], reverse=True)[:5])

    save_cluster_ply(args.infile, hdb.labels_, outdir=outdir)
    save_individual_clusters(str(outdir/'clusters.ply'), outdir=outdir)

    # Write output .ply files
    #save_centroid_ply(cluster_centroids, filename=f'{str(outdir / "qb_centroids.ply")}')
    #save_cluster_ply(streamlines, cluster_ids, filename=f'{str(outdir / "clusters.ply")}')
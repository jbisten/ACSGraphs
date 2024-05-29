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
sys.path.append(os.path.abspath("/home/justus/codebase/ACSGraphs/clustering/utils/"))
from plyio import save_centroid_ply, save_cluster_ply, save_individual_clusters, load_ply_file  

if __name__ == "__main__":
    
    # Argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Input file in .ply format")
    parser.add_argument("-o", "--outdir", help="Output directory")
    
    args = parser.parse_args()
    outdir = Path(args.outdir)
    infile = Path(args.infile)

    vertices, streamlines, end_ids, sub_ids, side_ids, vertex_features = load_ply_file(args.infile) 

    #################
    # Quickclustering
    #################
    qb = QuickBundles(threshold=33.)
    clusters = qb.cluster(streamlines)

    print(f"Found {len(clusters)} clusters")
    cluster_ids=np.zeros(len(streamlines))
    for n,c in enumerate(clusters):
       for i in c.indices:
           cluster_ids[i] = n


    # Write output .ply files
    # save_centroid_ply(cluster_centroids, filename=f'{str(outdir / "qb_centroids.ply")}')
    save_cluster_ply(infile, cluster_ids, filename=f'{str(outdir / "clusters.ply")}')
    save_individual_clusters(str(outdir/'clusters.ply'), outdir=outdir)
            

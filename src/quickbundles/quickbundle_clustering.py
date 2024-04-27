import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path
import nibabel as nib
from bonndit.utils.tck_io import Tck
from nibabel.streamlines import ArraySequence, save
from dipy.segment.clustering import QuickBundles


if __name__ == "__main__":
    
    # Argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Input file in .tck format")
    parser.add_argument("-o", "--outfile", help="Output path for file in .tck format")
    
    args = parser.parse_args()

    streamlines = nib.streamlines.load(args.infile)
    streamlines = streamlines.streamlines

    print(len(streamlines), type(streamlines))

    #################
    # Quickclustering
    #################
    qb = QuickBundles(threshold=6.)
    clusters = qb.cluster(streamlines)


    print(f"Found {len(clusters)} clusters")
    print('>>>>>>>>>>>>>>>>>')
    print("Cluster sizes:", [len(c) for c in clusters])
    print("Small clusters:", clusters < 10)
    print("Streamlines indices of the first cluster:\n", clusters[0].indices)
    print("Centroid of the last cluster:\n", clusters[-1].centroid)

    # TODO: Save file with all centroid streamlines to tck file 
    tck = Tck(args.outfile)
    tck.force = True
    tck.write({})
    [tck.append(cluster.centroid[:], None) for cluster in clusters]
    tck.close()

            

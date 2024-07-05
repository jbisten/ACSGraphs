import torch
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds
import os
import sys
import argparse  # Import argparse for command-line parsing
import nibabel as nib  # Import nibabel for handling neuroimaging files
import numpy as np  # Import numpy for numerical operations
from pathlib import Path
from plyfile import PlyData, PlyElement
import random
from clusterIO import load_tck, load_hdf5
from kmeans_pytorch import kmeans
import h5py
from bonndit.utils.tck_io import Tck

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infiles', type=str, action='append', help='input .tck files for processing')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    parser.add_argument('-q', '--qc', type=str, help='Output directory for quality control!')
    args = parser.parse_args()  # Parse command line arguments

    if args.infiles == None:
        print('ICP data already available, exiting . . .')
        sys.exit()


    qc_dir = Path(args.qc)

    # Load reference
    ref_vertices, ref_streamlines = load_tck(args.infiles[0])

    # Initialize
    n_streamlines, n_points, n_dims = np.array(ref_streamlines).shape 

    # kmeans for picking representatives
    #ref_cluster_ids, _ = kmeans(X=torch.from_numpy(ref_streamlines).reshape(ref_streamlines.shape[0], -1), num_clusters=100, distance='euclidean', device=torch.device('cuda:0'))
    #ref_vertices = np.vstack([np.mean(ref_streamlines[ref_cluster_ids == cluster_id], axis=0) for cluster_id in range(100)]).reshape(-1, n_dims)
    #print(ref_vertices.shape)

    # Write ref .tck-file, not necessary as this stays as it is
    tck = Tck(str(args.infiles[0]).replace('_mni_', '_mni_icp_'))
    tck.force = True
    tck.write({})
    [tck.append(s, None) for s in ref_streamlines]
    tck.close()

    # Picking random representatives 
    ref_indices = np.random.choice(ref_streamlines.shape[0], 100, replace=False)
    ref_vertices = np.vstack(ref_streamlines[ref_indices])
    ref_pc = Pointclouds(points=torch.tensor(ref_vertices, dtype=torch.float32).unsqueeze(0).to('cuda:0'))

        
    # Write quality control .tck-file
    qc_indices = np.random.choice(ref_streamlines.shape[0], 10000, replace=True)
    qc_streamlines = ref_streamlines[qc_indices] 
    tck = Tck(str(qc_dir / Path(args.infiles[0]).parts[-1]).replace('_mni_', '_mni_icp_qc_'))
    tck.force = True
    tck.write({})
    [tck.append(s, None) for s in qc_streamlines]
    tck.close()

    for infile in args.infiles[1:]:
        infile = Path(infile)

        assert infile.suffix == '.tck', "file is not a .tck-file"

        # Load vertices and streamlines
        src_vertices, src_streamlines = load_tck(infile)

        if 'rh' in infile.stem:  # Check if the tract is right-sided
            src_streamlines[:, :, 0] = -src_streamlines[:, :, 0]  # Flip x-coordinates

        # Randomly select 100 unique streamlines
        src_indices = np.random.choice(src_streamlines.shape[0], 100, replace=False)
        src_vertices = np.vstack(src_streamlines[src_indices])
        src_end_ids = np.cumsum(np.array([len(s) for s in src_streamlines]))

        # hallo justus  tsrc_cluster_ids, _ = kmeans(X=torch.from_numpy(src_streamlines).reshape(src_streamlines.shape[0], -1), num_clusters=100, distance='euclidean', device=torch.device('cuda:0'))
        # src_vertices = np.vstack([np.mean(src_streamlines[src_cluster_ids == cluster_id], axis=0) for cluster_id in range(100)]).reshape(-1, n_dims)
    
        src_pc = Pointclouds(points=torch.tensor(np.vstack(src_vertices), dtype=torch.float32).unsqueeze(0).to('cuda:0'))

        # Run the iterative closest point (ICP) algorithm
        print(f'Loaded {len(src_streamlines)} streamlines . . .')
        result = iterative_closest_point(src_pc, ref_pc)
        print('ICP Done! Applying transformation . . .')
        R, T, s = result.RTs.R[0].cpu().numpy(), result.RTs.T[0].cpu().numpy(), result.RTs.s[0].cpu().numpy()



        # Align streamlines to ref
        aligned_source_points = np.matmul(np.vstack(src_streamlines), R) + T
        aligned_streamlines = np.array([aligned_source_points[(e-n_points):e] for e in src_end_ids])

        # Write .tck-file
        tck = Tck(str(infile).replace('_mni_', '_mni_icp_'))
        tck.force = True
        tck.write({})
        [tck.append(s, None) for s in aligned_streamlines]
        tck.close()


        # Write quality control .tck-file
        qc_indices = np.random.choice(aligned_streamlines.shape[0], 10000, replace=True)
        qc_streamlines = aligned_streamlines[qc_indices] 
        tck = Tck(str(qc_dir / Path(infile).parts[-1]).replace('_mni_', '_mni_icp_qc_'))
        tck.force = True
        tck.write({})
        [tck.append(s, None) for s in qc_streamlines]
        tck.close()


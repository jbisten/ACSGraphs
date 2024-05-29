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
from plyio import load_ply_file
from kmeans_pytorch import kmeans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infiles', type=str, action='append', help='input .ply files for processing')
    parser.add_argument('-o', '--outfile', type=str, help='output path for the .ply file')
    parser.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    parser.add_argument('-n', '--n_supports', default=100, help='Number of points to which each streamline is resampled')
    
    args = parser.parse_args()  # Parse command line arguments
    outfile = Path(args.outfile)  # Set the output filename from argument
    n_supports = args.n_supports

    vertices = []
    sub_ids = []
    end_indices = []
    side_ids = []
    vertex_features = {}

    # Load reference
    ref_vertices, ref_streamlines, ref_end_ids, ref_sub_ids, ref_side_ids, ref_vertex_features = load_ply_file(args.infiles[0])
    
    # Initialize
    n_streamlines, n_points, n_dims = np.array(ref_streamlines).shape   

    # Already append ref data
    vertices.append(ref_vertices)
    sub_ids.append(ref_sub_ids)
    side_ids.append(ref_side_ids)
    end_indices.append(ref_end_ids)

    # Initialize end index offset to length of reference end indices 
    end_idx_offset = ref_vertices.shape[0]

    # kmeans for picking representatives
    print('Running kmeans for finding representatives')
    ref_cluster_ids, _ = kmeans(X=torch.from_numpy(ref_streamlines).reshape(ref_streamlines.shape[0], -1), num_clusters=100, distance='euclidean', device=torch.device('cuda:0'))
    ref_vertices = np.vstack([np.mean(ref_streamlines[ref_cluster_ids == cluster_id], axis=0) for cluster_id in range(100)]).reshape(-1, n_dims)
    
    print(ref_vertices.shape)

    # Picking random representatives 
    #ref_indices = np.random.choice(ref_streamlines.shape[0], 10, replace=False)
    #ref_vertices = np.vstack(ref_streamlines[ref_indices])
    
    ref_pc = Pointclouds(points=torch.tensor(ref_vertices, dtype=torch.float32).unsqueeze(0).to('cuda:0'))

    # Initializing vertex features as those of the reference
    vertex_features = ref_vertex_features

    for ply_in in args.infiles[1:]:
        ply_in = Path(ply_in)
        assert ply_in.suffix == '.ply', "file is not a .ply-file"

        # Load vertices and streamlines
        src_vertices, src_streamlines, src_end_ids, src_sub_ids, src_side_ids, src_vertex_features = load_ply_file(ply_in)
        if 'rh' in ply_in.stem:  # Check if the tract is right-sided
            src_streamlines[:, :, 0] = -src_streamlines[:, :, 0]  # Flip x-coordinates

        # Randomly select 100 unique streamlines
        # src_indices = np.random.choice(src_streamlines.shape[0], 10, replace=False) 
 
        print('Running kmeans for finding representatives: Src file edition')
        src_cluster_ids, _ = kmeans(X=torch.from_numpy(src_streamlines).reshape(src_streamlines.shape[0], -1), num_clusters=100, distance='euclidean', device=torch.device('cuda:0'))
        src_vertices = np.vstack([np.mean(src_streamlines[src_cluster_ids == cluster_id], axis=0) for cluster_id in range(100)]).reshape(-1, n_dims)
    
        src_pc = Pointclouds(points=torch.tensor(np.vstack(src_vertices), dtype=torch.float32).unsqueeze(0).to('cuda:0'))


        # Run the iterative closest point (ICP) algorithm
        result = iterative_closest_point(src_pc, ref_pc)
        R, T, s = result.RTs.R[0].cpu().numpy(), result.RTs.T[0].cpu().numpy(), result.RTs.s[0].cpu().numpy()

        # Align streamlines to ref
        aligned_source_points = np.matmul(np.vstack(src_streamlines), R) + T
        vertices.append(aligned_source_points)
        
        # Add new end indices with the addition of the offset
        end_indices.append(src_end_ids + end_idx_offset)
        n_streamlines += len(src_end_ids)
        end_idx_offset += aligned_source_points.shape[0]
       
        # Add subids
        sub_ids.append(src_sub_ids)
        side_ids.append(src_side_ids)

        # Extract and add vertex data excluding x, y, z
        src_vertex_feature_names = src_vertex_features.keys()
        for feature in src_vertex_feature_names:
            if feature not in ['x', 'y', 'z']:
                if feature not in vertex_features:
                    vertex_features[feature] = []
                vertex_features[feature].append(np.array(src_vertex_features[feature]).squeeze(0))


    # Set datatypes, assign data, and write to .ply file
    vertices = np.vstack(vertices)
    end_indices = np.hstack(end_indices)
    sub_ids = np.hstack(sub_ids)
    side_ids = np.hstack(side_ids)
    
    # Concatenate all vertex features
    for feature in vertex_features.keys():
        vertex_features[feature] = np.concatenate(vertex_features[feature])


    vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + [(feature, 'f4') for feature in vertex_features.keys()]
    fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4'), ('sideid', 'i4')]

    ply_vertices = np.empty(vertices.shape[0], dtype=vertex_dtypes)
    ply_vertices['x'] = vertices[:, 0]
    ply_vertices['y'] = vertices[:, 1]
    ply_vertices['z'] = vertices[:, 2]
    for feature in vertex_features.keys():
        ply_vertices[feature] = vertex_features[feature]

    ply_fibers = np.empty(n_streamlines, dtype=fiber_dtypes)
    ply_fibers['endindex'] = end_indices
    ply_fibers['subid'] = sub_ids 
    ply_fibers['sideid'] = side_ids 
   
    vertices = PlyElement.describe(ply_vertices, 'vertices') 
    fibers = PlyElement.describe(ply_fibers, 'fiber')

    PlyData([vertices, fibers], text=False).write(outfile)
    #print(f'Concatenated mni-warped .ply file with features to {outfile}')
    #print(f'Wrote {os.path.getsize(outfile)/1000000} Mb')

    

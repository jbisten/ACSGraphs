
from pathlib import Path
import argparse
from plyfile import PlyData, PlyElement
import nibabel as nib
import numpy as np
import os
import h5py
from bonndit.utils.tck_io import Tck

def save_feature_hdf5(vertex_features, end_indices, sub_indices, cluster_ids, outfile):
    pass

def save_subject_cluster_plys(vertices, vertex_features, end_indices, sub_indices, cluster_ids, outdir):
    # Get new vertices
    vertices = np.vstack([plydata['vertices'][field] for field in ['x', 'y', 'z']]).T
        
    # Add new end indices with the addition of the offset
    end_indices = plydata['fiber']['endindex']
    n = len(plydata['fiber']['endindex'])
    
    # Get subids
    sub_ids = plydata['fiber']['subid']
    side_ids = plydata['fiber']['sideid']

    # Extract vertex features from file     
    vertex_features = plydata['vertices'].data.dtype.names
    vertex_features = {feature: plydata['vertices'][feature] for feature in vertex_features if feature not in ['x', 'y', 'z']}

    # Set datatypes, assign data, and write to .ply file
    vertices = np.vstack(vertices)
    end_indices = np.array(end_indices)
    sub_ids = np.array(sub_ids)

    vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + [(feature, 'f4') for feature in vertex_features.keys()]
    fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4'), ('sideid', 'i4'), ('clusterid', 'i4')]

    ply_vertices = np.empty(vertices.shape[0], dtype=vertex_dtypes)
    ply_vertices['x'] = vertices[:, 0]
    ply_vertices['y'] = vertices[:, 1]
    ply_vertices['z'] = vertices[:, 2]
    for feature in vertex_features.keys():
        ply_vertices[feature] = vertex_features[feature]

    ply_fibers = np.empty(n, dtype=fiber_dtypes)
    ply_fibers['endindex'] = end_indices
    ply_fibers['subid'] = sub_ids
    ply_fibers['sideid'] = side_ids 
    ply_fibers['clusterid'] = cluster_ids 

    vertices = PlyElement.describe(ply_vertices, 'vertices') 
    fibers = PlyElement.describe(ply_fibers, 'fiber')

    PlyData([vertices, fibers], text=False).write(str(outdir / 'clusters.ply'))
    print(f'Cluster .ply file written to {str(outdir / "clusters.ply")}')

def save_cluster_representatives(streamlines, k, cluster_labels, outdir, n=10000):
    for cluster_id in range(k):
        # Get indices of streamlines in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        # If there are fewer than 1000 streamlines in the cluster, sample with replacement
        if len(cluster_indices) < n:
            sampled_indices = np.random.choice(cluster_indices, n, replace=True)
        else:
            sampled_indices = np.random.choice(cluster_indices, n, replace=False)
        
        # Get the sampled streamlines
        sampled_streamlines = streamlines[sampled_indices] 

        # Write .tck-file
        tck = Tck(str((outdir / f'cluster_{cluster_id}_representatives.tck')))
        tck.force = True
        tck.write({})
        [tck.append(s, None) for s in sampled_streamlines]
        tck.close()


def load_ply_file(filepath):
    """
    Load a PLY file and extract the vertices and fiber indices.

    Parameters:
    - filepath (str): Path to the PLY file.

    Returns:
    - vertices (np.ndarray): Array of vertices.
    - streamlines (list): List of streamlines.
    """
    plydata = PlyData.read(filepath)
    vertex_features = {}    
    
    # Extract and add vertex data excluding x, y, z
    ply_vertex_features = plydata['vertices'].data.dtype.names
    for feature in ply_vertex_features:
        if feature not in ['x', 'y', 'z']:
            if feature not in vertex_features:
                vertex_features[feature] = []
            vertex_features[feature].append(plydata['vertices'][feature])

    # Get streamlines level annotations 
    end_ids = plydata['fiber']['endindex']
    sub_ids = plydata['fiber']['subid']
    side_ids = plydata['fiber']['sideid']

    # Streamlines and vertices
    vertices = np.vstack([plydata['vertices']['x'], plydata['vertices']['y'], plydata['vertices']['z']]).T
    streamlines = np.array([vertices[start:end] for start, end in zip([0] + end_ids[:-1].tolist(), end_ids)])

    return vertices, streamlines, end_ids, sub_ids, side_ids, vertex_features


def load_tck(filepath):
    """
    Load a PLY file and extract the vertices and fiber indices.

    Parameters:
    - filepath (str): Path to the PLY file.

    Returns:
    - vertices (np.ndarray): Array of vertices.
    - streamlines (list): List of streamlines.
    """

    # Read streamline data
    tck = nib.streamlines.load(filepath)
    streamlines = np.array(tck.streamlines)
    vertices = np.vstack(streamlines)  

    return vertices, streamlines

def load_hdf5(filepath):
    """
    Reads the meta-information and datasets from an HDF5 file.

    Parameters:
    - filepath (str or Path): Path to the HDF5 file.

    Returns:
    - dict: A dictionary containing the meta-information and datasets.
    """
    meta_info = {}

    with h5py.File(filepath, 'r') as f:
        # Read attributes
        meta_info['n_streamlines'] = f.attrs.get('n_streamlines', 0)
        meta_info['n_supports'] = f.attrs.get('n_supports', 0)
        meta_info['sub_ids'] = f.attrs.get('sub_id',  0)
        meta_info['side_ids'] = f.attrs.get('side_id', 0)

        # Read mean values of the features
        for key in ['fa', 'ad', 'rd', 'md', 'l2', 'l3']:
            mean_key = f'mean_{key}'
            meta_info[mean_key] = f.attrs.get(mean_key, None)
        
        # Read datasets
        for key in ['fa', 'ad', 'rd', 'md', 'l2', 'l3']:
            if key in f:
                meta_info[key] = f[key][:]
    
    return meta_info

def save_subject_cluster_plys(vertices, vertex_features, end_indices, sub_indices, cluster_ids, outdir):
    # Get new vertices
    vertices = np.vstack([plydata['vertices'][field] for field in ['x', 'y', 'z']]).T
        
    # Add new end indices with the addition of the offset
    end_indices = plydata['fiber']['endindex']
    n = len(plydata['fiber']['endindex'])
    
    # Get subids
    sub_ids = plydata['fiber']['subid']
    side_ids = plydata['fiber']['sideid']

    # Extract vertex features from file     
    vertex_features = plydata['vertices'].data.dtype.names
    vertex_features = {feature: plydata['vertices'][feature] for feature in vertex_features if feature not in ['x', 'y', 'z']}

    # Set datatypes, assign data, and write to .ply file
    vertices = np.vstack(vertices)
    end_indices = np.array(end_indices)
    sub_ids = np.array(sub_ids)

    vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + [(feature, 'f4') for feature in vertex_features.keys()]
    fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4'), ('sideid', 'i4'), ('clusterid', 'i4')]

    ply_vertices = np.empty(vertices.shape[0], dtype=vertex_dtypes)
    ply_vertices['x'] = vertices[:, 0]
    ply_vertices['y'] = vertices[:, 1]
    ply_vertices['z'] = vertices[:, 2]
    for feature in vertex_features.keys():
        ply_vertices[feature] = vertex_features[feature]

    ply_fibers = np.empty(n, dtype=fiber_dtypes)
    ply_fibers['endindex'] = end_indices
    ply_fibers['subid'] = sub_ids
    ply_fibers['sideid'] = side_ids 
    ply_fibers['clusterid'] = cluster_ids 

    vertices = PlyElement.describe(ply_vertices, 'vertices') 
    fibers = PlyElement.describe(ply_fibers, 'fiber')

    PlyData([vertices, fibers], text=False).write(str(outdir / 'clusters.ply'))
    print(f'Cluster .ply file written to {str(outdir / "clusters.ply")}')
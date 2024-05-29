
from pathlib import Path
import argparse
from plyfile import PlyData, PlyElement
import nibabel as nib
import numpy as np
import os 

def save_centroid_ply(streamlines, filename):
    # Ensure streamlines is a numpy array
    n = streamlines.shape[0]
    streamlines = streamlines.numpy()
    end_indices = np.cumsum(np.array([len(s) for s in streamlines]))
    streamlines = np.vstack(streamlines)
    print(end_indices) 

    dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    ply_streamlines = np.empty(streamlines.shape[0], dtype=dtypes)
    
    ply_streamlines['x'] = streamlines[:, 0]
    ply_streamlines['y'] = streamlines[:, 1]
    ply_streamlines['z'] = streamlines[:, 2]
    
    tracks = PlyElement.describe(ply_streamlines, 'vertices') 
    endindex = PlyElement.describe(np.array(end_indices, dtype=[('endindex', 'i4')]), 'fiber')
    
    PlyData([tracks, endindex]).write(filename)
    print(f'Saved {n} streamlines to {filename} . . .')

def save_cluster_ply(plyfile, cluster_ids, outdir):
    ply_in = Path(plyfile)
    outdir = Path(outdir)
    assert ply_in.suffix == '.ply', "file is not a .ply-file"

    # Read file
    plydata = PlyData.read(ply_in)
    
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

def save_individual_clusters(plyfile, outdir):
    ply_in = Path(plyfile)
    outdir = Path(outdir)
    assert ply_in.suffix == '.ply', "file is not a .ply-file"

    # Read file
    plydata = PlyData.read(ply_in)
    
    cluster_ids = plydata['fiber']['clusterid']

    # Get unique clusters
    clusters = np.unique(cluster_ids)

    # Get global end indicies and subids
    end_indices = plydata['fiber']['endindex']
    sub_ids = plydata['fiber']['subid']
    side_ids = plydata['fiber']['sideid']
    vertices = plydata['vertices']

    for c in clusters:
        # Determine streamline level descriptors
        cluster_indices = np.where(cluster_ids == c)[0]
        cluster_sub_ids = sub_ids[cluster_indices]
        cluster_side_ids = side_ids[cluster_indices]
       
        n = len(cluster_indices)

        # Get vertex level descriptors for the cluster
        cluster_vertex_indices = []
        cluster_end_ids = []      
        i = 0 
        for idx in cluster_indices:
            start_idx = end_indices[idx-1] if idx > 0 else 0
            end_idx = end_indices[idx]
            cluster_vertex_indices.extend(range(start_idx, end_idx))
            i += end_idx-start_idx
            cluster_end_ids.append(i)


        cluster_end_indices = np.array(cluster_end_ids)

        # Ensure indices are within bounds
        cluster_vertex_indices = np.array(cluster_vertex_indices)
        cluster_vertices = vertices[cluster_vertex_indices]

        # Setup cluster specific vertex features and dtypes
        cluster_vertex_features = {feature: cluster_vertices[feature] for feature in cluster_vertices.dtype.names if feature not in ['x', 'y', 'z']}
        
        # Set datatypes, assign data, and write to .ply file
        cluster_vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + [(feature, 'f4') for feature in cluster_vertex_features.keys()]
        cluster_fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4'), ('sideid', 'i4'), ('clusterid', 'i4')]

        ply_vertices = np.empty(cluster_vertices.shape[0], dtype=cluster_vertex_dtypes)
        ply_vertices['x'] = cluster_vertices['x']
        ply_vertices['y'] = cluster_vertices['y']
        ply_vertices['z'] = cluster_vertices['z']
        for feature in cluster_vertex_features.keys():
            ply_vertices[feature] = cluster_vertex_features[feature]

        ply_fibers = np.empty(n, dtype=cluster_fiber_dtypes)
        ply_fibers['endindex'] = cluster_end_ids
        ply_fibers['subid'] = cluster_sub_ids
        ply_fibers['sideid'] = cluster_side_ids 
        ply_fibers['clusterid'] = cluster_ids[cluster_indices] 

        vertices_element = PlyElement.describe(ply_vertices, 'vertices') 
        fibers_element = PlyElement.describe(ply_fibers, 'fiber')

        PlyData([vertices_element, fibers_element], text=False).write(str(outdir / f'cluster_{c}.ply'))
        print(f'Saving Cluster {c}: {n} streamlines, {os.path.getsize(str(outdir / f"cluster_{c}.ply"))/1000000} Mb')

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
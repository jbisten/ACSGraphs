
from pathlib import Path
import argparse
from plyfile import PlyData, PlyElement
import nibabel as nib
import numpy as np
from bonndit.utils.tck_io import Tck 
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

def save_cluster_ply(streamlines, sub_ids, cluster_ids, filename):
    n = streamlines.shape[0]
    streamlines = streamlines.numpy()
    end_indices = np.cumsum(np.array([len(s) for s in streamlines]))
    streamlines = np.vstack(streamlines)

    vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4'), ('clusterid', 'i4')]

    ply_vertices = np.empty(streamlines.shape[0], dtype=vertex_dtypes)
    ply_vertices['x'] = streamlines[:, 0]
    ply_vertices['y'] = streamlines[:, 1]
    ply_vertices['z'] = streamlines[:, 2]

    
    ply_fibers = np.empty(n, dtype=fiber_dtypes)
    ply_fibers['endindex'] = end_indices 
    ply_fibers['subid'] = sub_ids
    ply_fibers['clusterid'] = cluster_ids

    vertices = PlyElement.describe(ply_vertices, 'vertices') 
    fibers = PlyElement.describe(ply_fibers, 'fiber')

    PlyData([vertices, fibers], text=True).write(filename)
    print(f'Wrote .ply file: {os.path.getsize(filename)/1000000} Mb')
   
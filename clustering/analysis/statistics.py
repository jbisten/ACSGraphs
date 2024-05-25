import torch
import numpy as np
from kmeans_pytorch import kmeans
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns 

class Cluster():
    def __init__(self, k, plydata):
        self.k = k 
        cluster_labels = plydata['fiber']['clusterid']
        
        # Get global end indicies and subids
        end_indices = plydata['fiber']['endindex']
        sub_ids = plydata['fiber']['subid']
        vertices = plydata['vertices']

        # Determine streamline level descriptors
        cluster_indices = np.where(cluster_labels == k)[0]
        self.sub_ids = sub_ids[cluster_indices]
        
        # Number of streamlines in this cluster
        self.n_streamlines = len(cluster_indices)

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

        self.end_indices = np.array(cluster_end_ids)

        # Ensure indices are within bounds
        cluster_vertex_indices = np.array(cluster_vertex_indices)
        self.vertices = vertices[cluster_vertex_indices]
        _, self.subject_counts = np.unique(self.sub_ids, return_counts=True) 

class ClusterPly():
    def __init__(self, plyfile):
        ply_in = Path(plyfile)
        assert ply_in.suffix == '.ply', "file is not a .ply-file"
        self.plydata = PlyData.read(ply_in)
        self.cluster_labels = self.plydata['fiber']['clusterid']
        self.clusters = [Cluster(k, self.plydata) for k in np.unique(self.cluster_labels)]

        # Get global end indicies and subids
        self.end_indices = self.plydata['fiber']['endindex']
        self.sub_ids = self.plydata['fiber']['subid']
        self.vertices = self.plydata['vertices']
       
    def get_support(self, plot=False):
        for C in self.clusters:
            print(f'Cluster{C.k}, support for each subject {C.subject_counts}')
            

    def get_cluster_diff(self, param='fa', plot=False):
        cluster_data = []
        for C in self.clusters:
            if param in C.vertices.dtype.names:
                streamline_means = []
                start_idx = 0
                for end_idx in C.end_indices:
                    streamline_means.append(np.mean(C.vertices[param][start_idx:end_idx]))
                    start_idx = end_idx

                mean_value = np.mean(streamline_means)
                print(f'Cluster {C.k}: mean {param} = {mean_value}')
                cluster_data.append(pd.DataFrame({
                    'Cluster': [C.k] * len(streamline_means),
                    param: streamline_means
                }))
            else:
                print(f'Parameter {param} not found in vertices data.')

        if plot and cluster_data:
            # Concatenate all data for plotting
            plot_data = pd.concat(cluster_data)
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Cluster', y=param, data=plot_data)
            plt.title(f'Boxplot of {param} across clusters')
            plt.show()

    def get_cluster_profiles(self):
        pass

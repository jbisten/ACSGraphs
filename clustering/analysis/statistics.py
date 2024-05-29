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
from scipy.stats import ttest_ind

class Cluster():
    def __init__(self, k, plydata, participants_df):
        self.k = k 
       
        # Get global end indicies and subids
        cluster_labels = plydata['fiber']['clusterid']
        end_indices = plydata['fiber']['endindex']
        sub_ids = plydata['fiber']['subid']
        vertices = plydata['vertices']
        

        # Determine streamline level descriptors
        self.cluster_indices = np.where(cluster_labels == k)[0]
        self.sub_ids = sub_ids[self.cluster_indices]
        self.n_streamlines = len(self.cluster_indices)

        # Get vertex level descriptors for the cluster
        cluster_vertex_indices = []
        cluster_end_ids = []      
        i = 0 
        for idx in self.cluster_indices:
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
    def __init__(self, plyfile, participants_tsv):
        ply_in = Path(plyfile)
        assert ply_in.suffix == '.ply', "file is not a .ply-file"
        self.plydata = PlyData.read(ply_in)
        self.cluster_labels = self.plydata['fiber']['clusterid']
        self.participants_df = pd.read_csv(participants_tsv, sep='\t')
        self.clusters = [Cluster(k, self.plydata, self.participants_df) for k in np.unique(self.cluster_labels)]

        # Get global end indicies and subids
        self.end_indices = self.plydata['fiber']['endindex']
        self.sub_ids = self.plydata['fiber']['subid']
        self.vertices = self.plydata['vertices']
        self.side_ids = self.plydata['fiber']['sideid']
        self.demographic = [self.participants_df.loc[self.participants_df['participant_id'] == f'sub-{sub_id:05d}'] for sub_id in self.sub_ids]

    def get_support(self, plot=False):
        for C in self.clusters:
            print(f'Cluster{C.k}, support for each subject {C.subject_counts}, and number of included subject: {len(C.subject_counts)}')
            

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

    
    def get_cluster_group_diff(self, param='fa', plot=True):
        cluster_data_left = []
        cluster_data_right = []

        for C in self.clusters:
            if param in C.vertices.dtype.names:
                subject_means = {}
                start_idx = 0
                for end_idx, sub_id, side_id in zip(C.end_indices, C.sub_ids, self.plydata['fiber']['sideid']):
                    participant_id = f'sub-{sub_id:05d}'
                    group = self.participants_df.loc[self.participants_df['participant_id'] == participant_id, 'Group'].values[0]
                    
                    # Initialize the dictionary for new subjects
                    if participant_id not in subject_means:
                        subject_means[participant_id] = {'mean': [], 'group': group, 'side': side_id}
                    
                    subject_means[participant_id]['mean'].append(np.mean(C.vertices[param][start_idx:end_idx]))
                    start_idx = end_idx

                # Compute mean for each subject
                for participant_id, data in subject_means.items():
                    data['mean'] = np.mean(data['mean'])

                # Create DataFrame for this cluster
                cluster_df = pd.DataFrame({
                    'Cluster': [C.k] * len(subject_means),
                    param: [data['mean'] for data in subject_means.values()],
                    'Group': [data['group'] for data in subject_means.values()],
                    'Side': [data['side'] for data in subject_means.values()]
                })

                cluster_data_left.append(cluster_df[cluster_df['Side'] == 0])
                cluster_data_right.append(cluster_df[cluster_df['Side'] == 1])
            else:
                print(f'Parameter {param} not found in vertices data.')

        if plot and (cluster_data_left or cluster_data_right):
            if cluster_data_left:
                plot_data_left = pd.concat(cluster_data_left)
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='Cluster', y=param, hue='Group', data=plot_data_left)
                plt.title(f'Boxplot of {param} across clusters - Left Hemisphere')
                plt.show()

            if cluster_data_right:
                plot_data_right = pd.concat(cluster_data_right)
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='Cluster', y=param, hue='Group', data=plot_data_right)
                plt.title(f'Boxplot of {param} across clusters - Right Hemisphere')
                plt.show()

        # Perform t-tests
        for side, cluster_data in [('Left Hemisphere', cluster_data_left), ('Right Hemisphere', cluster_data_right)]:
            if cluster_data:
                for cluster_df in cluster_data:
                    if not cluster_df.empty:
                        cluster_id = cluster_df['Cluster'].iloc[0]
                        patients = cluster_df[cluster_df['Group'] == 'patient'][param]
                        controls = cluster_df[cluster_df['Group'] == 'control'][param]
                        t_stat, p_val = ttest_ind(patients, controls, equal_var=False)  # Welch's t-test
                        print(f'T-test results for {param} in Cluster {cluster_id} ({side}): t-stat = {t_stat:.3f}, p-val = {p_val:.3f}')
            


    def get_cluster_profiles(self):
        pass


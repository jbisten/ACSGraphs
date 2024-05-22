import torch
import numpy as np
from kmeans_pytorch import kmeans
from pathlib import Path
import nibabel as nib
    

def gpu_sc(streamlines, k):
 # Check if the input is a NumPy array and convert if necessary
    if isinstance(streamlines, np.ndarray):
        streamlines = torch.from_numpy(streamlines)

    streamlines = streamlines.reshape(streamlines.shape[0], -1)
    
    # kmeans
    cluster_ids, cluster_centers = kmeans(
                                        X=streamlines, 
                                        num_clusters=k, 
                                        distance='euclidean', 
                                        device=torch.device('cuda:0')
    )

    return cluster_ids, cluster_centers

def get_cluster_centroids(streamlines, k, labels):
    # Ensure streamlines is a PyTorch tensor
    if not isinstance(streamlines, torch.Tensor):
        streamlines = torch.tensor(streamlines, dtype=torch.float32)
    
    # Ensure labels is a PyTorch tensor
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    # Initialize tensor to store centroids
    centroids = torch.zeros(k, streamlines.size(1), streamlines.size(2))

    # Calculate the centroid for each cluster
    for i in range(k):
        # Find all streamlines in this cluster
        mask = (labels == i)
        cluster_streamlines = streamlines[mask]
        
        # Compute the mean across all streamlines in the cluster
        assert cluster_streamlines.shape[0] > 0, 'Empty cluster . . .'  # Check to avoid empty slice
        
        centroids[i] = cluster_streamlines.mean(dim=0)

    return centroids


def get_min_dist_streams(streamlines):
    # Convert numpy ndarray to PyTorch tensor if it's not already a tensor
    if not isinstance(streamlines, torch.Tensor):
        streamlines = torch.tensor(streamlines, dtype=torch.float32)

    # Select the first streamline as the reference
    ref_streamline = streamlines[0]

    # Compute distances to the non-flipped version of the reference
    dist_non_flipped = torch.norm(streamlines - ref_streamline, dim=2).sum(dim=1)

    # Compute distances to the flipped version of the reference
    ref_streamline_flipped = torch.flip(ref_streamline, dims=[0])
    dist_flipped = torch.norm(streamlines - ref_streamline_flipped, dim=2).sum(dim=1)

    # Determine the minimal distances and select accordingly
    min_dist, indices = torch.min(torch.stack([dist_non_flipped, dist_flipped], dim=1), dim=1)

    # Use these indices to flip streamlines where the flipped version was closer
    min_dist_streamlines = torch.where(indices.unsqueeze(1).unsqueeze(2).expand_as(streamlines) == 1,
                                       torch.flip(streamlines, dims=[1]),
                                       streamlines)
    


    return min_dist_streamlines



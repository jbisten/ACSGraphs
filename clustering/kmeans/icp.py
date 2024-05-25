import torch
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds

def icp(source_ply, target_ply, flip=False):


    


    # Convert to PyTorch3D Pointclouds structure
    source_pc = Pointclouds(points=source_vertices)
    target_pc = Pointclouds(points=target_vertices)

    # Run the iterative closest point (ICP) algorithm
    result = iterative_closest_point(source_pc, target_pc)

    # Extract the transformation matrix
    R, T = result.RTs[0]

    # Apply the transformation to the source points
    #aligned_source_points = torch.matmul(source_points, R.t()) + T

    # Print the transformation results
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    print("Aligned Source Points:\n", aligned_source_points)

    # Optionally, compute and print the alignment error
    alignment_error = torch.norm(target_points - aligned_source_points).mean().item()
    print("Alignment Error: ", alignment_error)

    # Save matrices to ants readable .mat file
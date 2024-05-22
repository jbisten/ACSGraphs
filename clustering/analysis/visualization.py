import torch
import numpy as np
from kmeans_pytorch import kmeans
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm

def load_cluster_ply(cluster_ply):
    plydata = PlyData.read(cluster_ply)
    vertices = np.vstack([plydata['vertices']['x'], plydata['vertices']['y'], plydata['vertices']['z']]).T
    end_indices = plydata['fiber']['endindex']
    sub_ids = plydata['fiber']['subid']
    cluster_ids = plydata['fiber']['clusterid']
    streamlines= np.array([vertices[i - 21:i] for i in end_indices]) 

    return streamlines, sub_ids, cluster_ids 

def get_tangent_and_plane(streamline, vertex_index):
    """
    Calculate the tangent vector and the equation of the plane at a given vertex index.

    Parameters:
    streamline (numpy.ndarray): Array of shape (21, 3) representing the streamline.
    vertex_index (int): The index of the vertex at which to calculate the tangent and plane.

    Returns:
    tuple: Tangent vector and plane equation coefficients.
    """
    if vertex_index < 0 or vertex_index >= streamline.shape[0]:
        raise ValueError("vertex_index out of bounds")

    coordinates = streamline
    if vertex_index == 0:
        tangent = coordinates[vertex_index + 1] - coordinates[vertex_index]
    elif vertex_index == streamline.shape[0] - 1:
        tangent = coordinates[vertex_index] - coordinates[vertex_index - 1]
    else:
        tangent = coordinates[vertex_index + 1] - coordinates[vertex_index - 1]

    tangent = tangent / np.linalg.norm(tangent)  # Normalize the tangent vector

    point = coordinates[vertex_index]
    normal = tangent

    # Using a symbolic representation for plane equation 'a(x - x0) + b(y - y0) + c(z - z0) = 0'
    a, b, c = normal
    x0, y0, z0 = point

    plane_equation = (a, b, c, - (a * x0 + b * y0 + c * z0))

    return tangent, plane_equation

def get_plane_intersection(streamlines, point_on_plane, plane_normal):
    """
    Find intersection points of streamlines with a specified plane.
    
    Args:
    streamlines (np.ndarray): Array of streamlines with shape (n, 21, 3).
    point_on_plane (np.ndarray): A point on the plane (3,).
    plane_normal (np.ndarray): Normal vector of the plane (3,).
    
    Returns:
    np.ndarray: Intersection points, one per streamline that intersects.
    """
    intersections = []
    for streamline in streamlines:
        for i in range(len(streamline) - 1):
            p1 = streamline[i]
            p2 = streamline[i + 1]
            # Calculate parameter t for the line intersection formula
            d = p2 - p1
            denom = np.dot(plane_normal, d)
            if abs(denom) > 1e-5:  # To avoid division by zero
                t = np.dot(plane_normal, point_on_plane - p1) / denom
                if 0 <= t <= 1:  # Check if the intersection point lies within the segment
                    intersection = p1 + t * d
                    intersections.append(intersection)
                    break  # Assuming only one intersection per streamline
    return np.array(intersections)

def get_cluster_mean_fa(cluster_ply):
     


def plot_cluster_intersections(cluster_ply, vertex_index, multisub=False):
    streamlines, sub_ids, cluster_ids = load_cluster_ply(cluster_ply) 
    center = np.mean(streamlines, axis=0)
    tangent, plane = get_tangent_and_plane(center, vertex_index)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2], color='red', label='Intersections')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


    # Make Screenshot from angle with vvi for background

    # Get tangent and plane

    # Get intersection points 

    # Plot intersecting points

def plot_slice_intersections(nifti_file, streamlines, cluster_ids, slice_idx, point_size, direction):
    # Load the NIfTI file
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    # Transform streamlines from world space to voxel (index) space
    inv_affine = np.linalg.inv(affine)
    streamlines_vox = [nib.affines.apply_affine(inv_affine, streamline) for streamline in streamlines]

    # Define the slice
    if direction == 'sagittal':
        slice_data = data[slice_idx, :, :]
    elif direction == 'coronal':
        slice_data = data[:, slice_idx, :]
    elif direction == 'axial':
        slice_data = data[:, :, slice_idx]
    else:
        raise ValueError("Invalid direction. Choose from 'sagittal', 'coronal', 'axial'.")

    # Plot the slice
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data.T, cmap='gray', origin='lower')
    
    # Define colors for each cluster
    num_clusters = len(np.unique(cluster_ids))
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))
    
    # Plot the streamline intersections with hue based on cluster IDs
    for streamline, cluster_id in zip(streamlines_vox, cluster_ids):
        color = colors[cluster_id]
        for point in streamline:
            if direction == 'sagittal' and round(point[0]) == slice_idx:
                plt.plot(point[1], point[2], 'o', color=color, markersize=point_size)
            elif direction == 'coronal' and round(point[1]) == slice_idx:
                plt.plot(point[0], point[2], 'o', color=color, markersize=point_size)
            elif direction == 'axial' and round(point[2]) == slice_idx:
                plt.plot(point[0], point[1], 'o', color=color, markersize=point_size)

    plt.title(f"{direction.capitalize()} Slice {slice_idx}")
    plt.xlabel('X-axis' if direction != 'sagittal' else 'Y-axis')
    plt.ylabel('Y-axis' if direction != 'axial' else 'Z-axis')
    plt.show()


import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
from bonndit.utils.tck_io import Tck
from nibabel.streamlines import ArraySequence, save
from plyfile import PlyElement, PlyData
import os 

def transform_streamlines(streamlines, transform, inv):
    transform = Path(transform)
    assert transform.is_file(), 'The specifiend transform file does not exist'
    if transform.suffix == '.mat':
        streamlines = np.hstack((streamlines, np.ones((streamlines.shape[0],1)))) 
        
        sitk_transform = sitk.ReadTransform(transform)
        affine = np.zeros((4,4))
        
        # All global transformations are of the form T(x) = A(x-c) + t + c
        A =  np.array(sitk_transform.GetMatrix()).reshape(3,3)
        c = np.array(sitk_transform.GetCenter())
        t = np.array(sitk_transform.GetTranslation())

        affine = np.eye(4)  # Initialize as identity matrix

        # Apply RAS flipping if needed
        rasflip = np.diag([-1, -1, 1, 1])  # Flip R and A coordinates, leave S and homogeneous coordinate unchanged

        # Calculate the adjusted affine transformation
        # Directly incorporate the center and translation without altering for RAS flipping
        affine[:3, :3] = A
        affine[:3, 3] = t + c - A.dot(c)

        # Apply RAS flipping to the affine matrix if needed
        affine = rasflip @ affine @ np.linalg.inv(rasflip)

        inverse = np.linalg.inv(affine)
       
        if inv == 1:
            transformed_streamlines = (inverse @ streamlines.T).T
        else:
            transformed_streamlines = (affine @ streamlines.T).T

        return transformed_streamlines[:, :3]
    
    elif transform.suffix == '.gz':
        # Transform to index space
        ref_nifti = nib.load(args.ref)
        affine = ref_nifti.affine
        inverse = np.linalg.inv(affine)
        index_streamlines = (inverse @ np.hstack((streamlines, np.ones((streamlines.shape[0],1)))).T).T[:, :3] 

        warp_data = load_ants_warpfield(transform)
        x, y, z = np.mgrid[0:warp_data.shape[0], 0:warp_data.shape[1], 0:warp_data.shape[2]] # The warpfield's grid coordinates
        interpolator = RegularGridInterpolator((x[:, 0, 0], y[0, :, 0], z[0, 0, :]), warp_data)
        displacements = interpolator(index_streamlines)
        warped_vertices = streamlines + displacements

        return warped_vertices

def load_ants_warpfield(warp_nii):
    warp_nii = nib.load(warp_nii)
    warp_data = warp_nii.get_fdata()
    warp_data = np.squeeze(warp_data, axis=3)
    return warp_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Input file in .tck or .ply format")
    parser.add_argument("-r", "--ref", help="Reference file to map streamlines into index space. Append '@0' for direct or '@1' for inverse transform. Example: -t file1,0 -t file2,1")
    parser.add_argument("-w", "--world", help="Reference file to map streamlines into the output space")
    parser.add_argument('-t', '--transform', action='append', help='Transforms, can be either affine matrices as .mat files or ANTs 5D warpfields. Append "@0" for direct or "@1" for inverse transform. Example: -t file1,0 -t file2,1') 
    parser.add_argument("-o", "--outfile", help="Output path for file in .tck format")
    parser.add_argument("-n", "--n_supports", default=100, help="Output path for file in .tck format")
  
    args = parser.parse_args()

    n_supports = args.n_supports
    outfile = Path(args.outfile)

    # Prepare streamlines
    infile = Path(args.infile)
    
    if infile.suffix == '.ply':
        # TODO: Read out any potential meta information 
        plydata = PlyData.read(infile)
        vertex_features = plydata['vertices'].data.dtype.names
        vertices = np.vstack([plydata['vertices'][field] for field in ['x', 'y', 'z']]).T
        end_indices = plydata['fiber']['endindex']
        sub_ids = plydata['fiber']['subid']
        streamlines = np.array([vertices[i - n_supports:i] for i in end_indices])
        n_streamlines = len(streamlines) 
        
        # Extract vertex data excluding x, y, z
        vertex_features = {feature: plydata['vertices'][feature] for feature in vertex_features if feature not in ['x', 'y', 'z']}

    elif infile.suffix == '.tck':
        streamlines = nib.streamlines.load(args.infile).streamlines
        n_streamlines = len(streamlines) 

    indices = []
    n = 0
    for sl in streamlines:
        start = n
        end = n + len(sl)
        n = end 
        indices.append((start, end))

    streamlines = np.vstack(streamlines)

    if args.transform:
        for transform in args.transform:
            transform, inv = transform.split('@')
            streamlines = transform_streamlines(streamlines, transform, int(inv))

    if outfile.suffix == '.tck':
        tck = Tck(str(outfile))
        tck.force = True
        tck.write({})
        [tck.append(streamlines[s:e], None) for s, e in indices]
        tck.close()

    elif outfile.suffix == '.ply':
        vertices = streamlines # our new, warped streamlines
     
        # Set datatypes, assign data, and write to .ply file
        vertex_dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + [(feature, 'f4') for feature in vertex_features.keys()]
        fiber_dtypes = [('endindex', 'i4'), ('subid', 'i4')]

        ply_vertices = np.empty(vertices.shape[0], dtype=vertex_dtypes)
        ply_vertices['x'] = vertices[:, 0]
        ply_vertices['y'] = vertices[:, 1]
        ply_vertices['z'] = vertices[:, 2]
        for feature in vertex_features.keys():
            ply_vertices[feature] = vertex_features[feature]

        ply_fibers = np.empty(n_streamlines, dtype=fiber_dtypes)
        ply_fibers['endindex'] = end_indices
        ply_fibers['subid'] = sub_ids 

        vertices = PlyElement.describe(ply_vertices, 'vertices') 
        fibers = PlyElement.describe(ply_fibers, 'fiber')

        PlyData([vertices, fibers], text=False).write(outfile)
        #print(f'Wrote mni warped .ply file with features to {outfile}')
        #print(f'Wrote {os.path.getsize(outfile)/1000000} Mb')




    

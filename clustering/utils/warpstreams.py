import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
from bonndit.utils.tck_io import Tck
from nibabel.streamlines import ArraySequence, save

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
        #translation =  (np.identity(3)-A) @ c + t  
        #rasflip = np.identity(3)
        #rasflip[0,0] = -1
        #rasflip[1,1] = -1 

        #translation = rasflip @ translation

        #affine[:3, :3] = np.linalg.inv(rasflip) @ A @ rasflip
        #affine[:3, 3] = translation
        #affine[3,3] = 1

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

    #if args.world:
    #    print('Transform to World Space', streamlines.shape)
    #    if Path(transform).suffix == ".gz":
    #        streamlines = np.hstack((streamlines, np.ones((streamlines.shape[0],1))))
    #    affine_mni = nib.load(args.world).affine
    #    #print(inverse)
    #    streamlines = affine_mni @ streamlines.T 
    #    streamlines = streamlines.T

        return warped_vertices

def load_ants_warpfield(warp_nii):
    warp_nii = nib.load(warp_nii)
    warp_data = warp_nii.get_fdata()
    warp_data = np.squeeze(warp_data, axis=3)
    return warp_data

def load_affine_mat():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Input file in .tck format")
    parser.add_argument("-r", "--ref", help="Reference file to map streamlines into index space. Append '@0' for direct or '@1' for inverse transform. Example: -t file1,0 -t file2,1")
    parser.add_argument("-w", "--world", help="Reference file to map streamlines into the output space")
    parser.add_argument('-t', '--transform', action='append', help='Transforms, can be either affine matrices as .mat files or ANTs 5D warpfields. Append "@0" for direct or "@1" for inverse transform. Example: -t file1,0 -t file2,1') 
    parser.add_argument("-o", "--outfile", help="Output path for file in .tck format")
   
    args = parser.parse_args()

    # Prepare streamlines
    streamlines = nib.streamlines.load(args.infile)
    indices = []
    n = 0
    for sl in streamlines.streamlines:
        start = n
        end = n + len(sl)
        n = end 
        indices.append((start, end))

    streamlines = np.vstack(streamlines.streamlines)

    if args.transform:
        for transform in args.transform:
            transform, inv = transform.split('@')
            streamlines = transform_streamlines(streamlines, transform, int(inv))

    tck = Tck(args.outfile)
    tck.force = True
    tck.write({})
    [tck.append(streamlines[s:e], None) for s, e in indices]
    tck.close()






    

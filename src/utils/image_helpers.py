import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler
import os


def load_and_preprocess_nifti(mri_path):
    """
    Load a MRI NIfTI file and standardize its intensities.

    Decodes the input path from a TensorFlow byte string, loads the NIfTI image, standardizes 
    non-zero voxel intensities to zero mean and unit variance, and adds a channel dimension.

    Parameters
    ----------
    mri_path : string or tf.Tensor
        String or tensorFlow byte-string tensor containing the file path to the NIfTI MRI image 
        (.nii or .nii.gz).

    Returns
    -------
    mri_img_data : np.ndarray, shape (H, W, D, 1), dtype float32
        Standardized MRI image array with an added channel dimension.
    """
    if hasattr(mri_path, "numpy"):
        mri_path = mri_path.numpy().decode("utf-8")
        
    mri_img = nib.load(mri_path)
    mri_img_data = mri_img.get_fdata()

    mri_img_data = standardize_image(mri_img_data)
    mri_img_data = np.expand_dims(mri_img_data, axis=-1).astype(np.float32)

    return mri_img_data

def standardize_image(image):
    """
    Standardize non-zero voxels of a 3D image to zero mean and unit variance.

    Zero voxels are treated as background and are left unchanged. If no non-zero voxels are 
    present, the image is returned unmodified.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, D)
        3D image array to be standardized.

    Returns
    -------
    image : np.ndarray, shape (H, W, D)
        Image array where non-zero voxels have been standardized to zero mean
    """
    nonzero_mask = (image != 0)
    nonzero_image = image[nonzero_mask]

    if nonzero_image.size > 0:
        nonzero_image = (nonzero_image - nonzero_image.mean()) / nonzero_image.std()
        image[nonzero_mask] = nonzero_image

    return image


def load_nifti(im_path):
    """
    Load a NIfTI image and add a channel dimension.

    Decodes the input path from a TensorFlow byte string, loads the NIfTI image, and adds a 
    trailing channel dimension.

    Parameters
    ----------
    im_path : tf.Tensor
        TensorFlow byte-string tensor containing the file path to the NIfTI image 
        (.nii or .nii.gz).

    Returns
    -------
    im_data : np.ndarray, shape (H, W, D, 1), dtype float32
        Image array with an added channel dimension.
    """
    
    if hasattr(im_path, "numpy"):
        im_path = im_path.numpy().decode("utf-8")
        
    im_img = nib.load(im_path)
    im_data = im_img.get_fdata()

    im_data = np.expand_dims(im_data, axis=-1).astype(np.float32)

    return im_data

def load_multiple_modalities_train(mri_path, taupet_path, mask_path, plasma_value, age_value, weights=1.0):
    """
    Load and prepare all modalities for one training sample.

    Loads the MRI (with standardization), tau-PET, and brain mask NIfTI files, and converts
    scalar clinical variables (plasma p-tau217, age, and sample weight) to TensorFlow tensors.

    Parameters
    ----------
    mri_path : tf.Tensor
        TensorFlow byte-string tensor with the file path to the MRI NIfTI image.
    taupet_path : tf.Tensor
        TensorFlow byte-string tensor with the file path to the tau-PET NIfTI image.
    mask_path : tf.Tensor
        TensorFlow byte-string tensor with the file path to the brain mask NIfTI image.
    plasma_value : float or tf.Tensor
        Scaled plasma p-tau217 value for the subject.
    age_value : float or tf.Tensor
        Scaled age value for the subject.
    weights : float, optional
        Sample weight for the training loss. Default is 1.0.

    Returns
    -------
    mri_img : tf.Tensor, shape (72, 90, 76, 1), dtype float32
        Standardized MRI image tensor.
    taupet_img : tf.Tensor, shape (72, 90, 76, 1), dtype float32
        Tau-PET image tensor.
    mask_img : tf.Tensor, shape (72, 90, 76, 1), dtype float32
        Brain mask image tensor.
    plasma_tensor : tf.Tensor, shape (1,), dtype float32
        Plasma p-tau217 value as a 1-D tensor.
    age_tensor : tf.Tensor, shape (1,), dtype float32
        Age value as a 1-D tensor.
    weights_tensor : tf.Tensor, shape (), dtype float32
        Sample weight as a scalar tensor.
    """
    mri_img = tf.py_function(func=load_and_preprocess_nifti, inp=[mri_path], Tout=tf.float32)
    taupet_img = tf.py_function(func=load_nifti, inp=[taupet_path], Tout=tf.float32)
    mask_img = tf.py_function(func=load_nifti, inp=[mask_path], Tout=tf.float32)

    # Set static shapes
    mri_img.set_shape([72, 90, 76, 1])
    taupet_img.set_shape([72, 90, 76, 1])
    mask_img.set_shape([72, 90, 76, 1])

    plasma_tensor = tf.convert_to_tensor(plasma_value, dtype=tf.float32)
    plasma_tensor = tf.reshape(plasma_tensor, [-1])

    age_tensor = tf.convert_to_tensor(age_value, dtype=tf.float32)
    age_tensor = tf.reshape(age_tensor, [-1])

    weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)

    return mri_img, taupet_img, mask_img, plasma_tensor, age_tensor, weights_tensor


def load_multiple_modalities_test(mri_path, mask_path, plasma_value, age_value, weights=1.0):
    """
    Load and prepare all modalities for one test sample.

    Loads the MRI (with standardization) and brain mask NIfTI files, and converts scalar 
    clinical variables (plasma p-tau217, age, and sample weight) to TensorFlow tensors. 
    Unlike the training loader, no tau-PET image is loaded.

    Parameters
    ----------
    mri_path : tf.Tensor
        TensorFlow byte-string tensor with the file path to the MRI NIfTI image.
    mask_path : tf.Tensor
        TensorFlow byte-string tensor with the file path to the brain mask NIfTI image.
    plasma_value : float or tf.Tensor
        Scaled plasma p-tau217 value for the subject.
    age_value : float or tf.Tensor
        Scaled age value for the subject.
    weights : float, optional
        Sample weight. Default is 1.0.

    Returns
    -------
    mri_img : tf.Tensor, shape (72, 90, 76, 1), dtype float32
        Standardized MRI image tensor.
    mask_img : tf.Tensor, shape (72, 90, 76, 1), dtype float32
        Brain mask image tensor.
    plasma_tensor : tf.Tensor, shape (1,), dtype float32
        Plasma p-tau217 value as a 1-D tensor.
    age_tensor : tf.Tensor, shape (1,), dtype float32
        Age value as a 1-D tensor.
    weights_tensor : tf.Tensor, shape (), dtype float32
        Sample weight as a scalar tensor.
    """
    mri_img = tf.py_function(func=load_and_preprocess_nifti, inp=[mri_path], Tout=tf.float32)
    mask_img = tf.py_function(func=load_nifti, inp=[mask_path], Tout=tf.float32)

    mri_img.set_shape([72, 90, 76, 1])
    mask_img.set_shape([72, 90, 76, 1])

    plasma_tensor = tf.convert_to_tensor(plasma_value, dtype=tf.float32)
    plasma_tensor = tf.reshape(plasma_tensor, [-1])

    age_tensor = tf.convert_to_tensor(age_value, dtype=tf.float32)
    age_tensor = tf.reshape(age_tensor, [-1])

    weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)

    return mri_img, mask_img, plasma_tensor, age_tensor, weights_tensor


def load_multiple_modalities_test2(mri_path, plasma_value, age_value, weights=1.0):
    """
    Load and prepare all modalities for one test sample.

    Loads the MRI (with standardization) and brain mask NIfTI files, and converts scalar 
    clinical variables (plasma p-tau217, age, and sample weight) to TensorFlow tensors. 
    Unlike the training loader, no tau-PET image is loaded.

    Parameters
    ----------
    mri_path : tf.Tensor
        TensorFlow byte-string tensor with the file path to the MRI NIfTI image.
    plasma_value : float or tf.Tensor
        Scaled plasma p-tau217 value for the subject.
    age_value : float or tf.Tensor
        Scaled age value for the subject.

    Returns
    -------
    mri_img : tf.Tensor, shape (72, 90, 76, 1), dtype float32
        Standardized MRI image tensor.
    plasma_tensor : tf.Tensor, shape (1,), dtype float32
        Plasma p-tau217 value as a 1-D tensor.
    age_tensor : tf.Tensor, shape (1,), dtype float32
        Age value as a 1-D tensor.
    """
    mri_img = tf.py_function(func=load_and_preprocess_nifti, inp=[mri_path], Tout=tf.float32)
    mri_img.set_shape([72, 90, 76, 1])

    plasma_tensor = tf.convert_to_tensor(plasma_value, dtype=tf.float32)
    plasma_tensor = tf.reshape(plasma_tensor, [-1])

    age_tensor = tf.convert_to_tensor(age_value, dtype=tf.float32)
    age_tensor = tf.reshape(age_tensor, [-1])

    return mri_img, plasma_tensor, age_tensor


def build_file_paths(df, mri_root, taupet_root, mask_root):
    """
    Build full file paths for all subjects in a dataframe.

    Constructs lists of absolute file paths for MRI, tau-PET, and brain mask images by 
    joining root directories with per-subject folder names and fixed filenames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``file_names`` column with subject-level folder names.
    mri_root : str
        Root directory containing per-subject MRI folders.
    taupet_root : str
        Root directory containing per-subject tau-PET folders.
    mask_root : str
        Root directory containing per-subject brain mask folders.

    Returns
    -------
    mri_paths : list of str
        Full file paths to MRI images (``jademarec.nii.gz``), one per subject.
    taupet_paths : list of str
        Full file paths to tau-PET images (``smoothed_image.nii.gz``), one per subject.
    mask_paths : list of str
        Full file paths to brain mask images (``jademarec.nii.gz``), one per subject.
    """
    mri_paths = []
    taupet_paths = []
    mask_paths = []

    for file_name in df["file_names"]:
        mri_paths.append(os.path.join(mri_root, file_name, "jademarec.nii.gz"))
        taupet_paths.append(os.path.join(taupet_root, file_name, "smoothed_image.nii.gz"))
        mask_paths.append(os.path.join(mask_root, file_name, "jademarec.nii.gz"))

    return mri_paths, taupet_paths, mask_paths


def generate_paths(file_name,mri_root,taupet_root,mask_root):
    """
    Generate MRI, tau-PET, and brain mask file paths for a single subject.

    Parameters
    ----------
    file_name : str
        Subject-level folder name used to locate the image files.
    mri_root : str
        Root directory containing per-subject MRI folders.
    taupet_root : str
        Root directory containing per-subject tau-PET folders.
    mask_root : str
        Root directory containing per-subject brain mask folders.

    Returns
    -------
    mri_path : str
        Full file path to the MRI image (``jademarec.nii.gz``).
    taupet_path : str
        Full file path to the tau-PET image (``smoothed_image.nii.gz``).
    mask_path : str
        Full file path to the brain mask image (``jademarec.nii.gz``).
    """
    mri_path = os.path.join(mri_root, file_name, "jademarec.nii.gz")
    taupet_path = os.path.join(taupet_root, file_name, "smoothed_image.nii.gz")
    mask_path = os.path.join(mask_root, file_name, "jademarec.nii.gz")
    return mri_path, taupet_path, mask_path
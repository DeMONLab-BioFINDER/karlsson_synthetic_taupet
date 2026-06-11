"""
Generate synthetic tau-PET scans for a new test dataset.
"""

# Imports
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from pretrained_models.unet_taupet_v1.unet3d_v1 import build_tau_pet_unet
import src.utils.image_helpers as ih


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
test_csv = "datasets/df_test.csv" #INSERT test dataset dir
mri_root = " " #INSERT MRI root

# U-Net weights from trained model
weights_path = "pretrained_models/unet_taupet_v1/weights_v1.hdf5"

fillna_age = 70.08492087077349 # mean age in training set
fillna_plasma = 0.34589147447495605 # mean plasma p-tau217 in training set

# scalers for z-scoring according to training set
scaler_age = joblib.load("pretrained_models/unet_taupet_v1/scaler_age.joblib")
scaler_plasma = joblib.load("pretrained_models/unet_taupet_v1/scaler_plasma.joblib")

prediction_output_dir = "pretrained_models/unet_taupet_v1/synthetic_test_scans"

if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def get_dataframe():
    """
    Load and preprocess dataframe for prediction.
    """
    #impute missing age and plasma p-tau217
    df_test = pd.read_csv(test_csv)
    
    df_test["age_imputed"] = df_test["age"].fillna(fillna_age)
    df_test["plasma_ptau217_imputed"] = df_test["plasma_ptau217"].fillna(fillna_plasma)

    # Scale plasma
    df_test["plasma_ptau217_imputed"] = scaler_plasma.transform(df_test[["plasma_ptau217_imputed"]])

    # Scale age
    df_test["age_imputed"] = scaler_age.transform(df_test[["age_imputed"]])
    
    return df_test


def build_test_dataset(df_test,batch_size=1):
    """
    Build the test dataset.
    """
    
    mri_test_paths = []
    for subject_id in df_test['file_names']:
        mri_path = os.path.join(mri_root,subject_id,"jademarec.nii.gz")
        mri_test_paths.append(mri_path)
    
    print("Test paths successfully read")

    plasma_values = df_test["plasma_ptau217_imputed"].values.astype(np.float32)
    age_values = df_test["age_imputed"].values.astype(np.float32)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (mri_test_paths, plasma_values, age_values)
    )

    test_dataset = test_dataset.map(
        lambda x1, x2, x3: ih.load_multiple_modalities_test2(x1, x2, x3),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Test dataset successfully created")
    return test_dataset


def save_and_process_batch(i, mri_test, plasma_value, age_value, model,filename):
    """
    Process one batch for prediction.
    """
    print("starting batch " + str(i))

    prediction = model([mri_test, plasma_value, age_value], training=False)
    image_pred = prediction[0, :, :, :, 0]
    
    nifti_image = nib.Nifti1Image(image_pred.numpy(), np.eye(4))
    print(filename)
    save_path = os.path.join(prediction_output_dir, filename + ".nii.gz")
    print(save_path)
    nib.save(nifti_image, save_path)

    print("done with batch " + str(i))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
df_test = get_dataframe()
test_dataset = build_test_dataset(df_test)

# Build model and load weights
model = build_tau_pet_unet(verbose=True)
model.load_weights(weights_path)

# Convert dataset to list for easier parallel processing
test_data_list = list(test_dataset)
indices = list(range(len(test_data_list)))

mri_tests = [data[0] for data in test_data_list]
plasma_tests = [data[1] for data in test_data_list]
age_tests = [data[2] for data in test_data_list]
filenames = list(df_test['file_names'])


def process_batch_wrapper(i, mri_test, plasma_value, age_value,filename):
    return save_and_process_batch(i, mri_test, plasma_value, age_value, model,filename)

with ThreadPoolExecutor(max_workers=1) as executor:
    results = executor.map(
        process_batch_wrapper,
        indices,
        mri_tests,
        plasma_tests,
        age_tests,
        filenames
    )
"""
Evaluation and inference script for synthetic tau-PET prediction.
"""

# Imports
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler
import joblib

from src.models.unet3d import build_tau_pet_unet
import src.utils.image_helpers as ih

# --------------------------------------------------
# PATHS
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.join(script_dir, "..")

# Datasets (relative to repo root)
data_dir     = os.path.join(repo_root, "datasets", "simulated_example")
train_csv    = os.path.join(data_dir, "simulated_example_df.csv")       # Replace with path for train csv
test_csv     = os.path.join(data_dir, "simulated_example_df.csv")       # Replace with path for test csv
mri_root     = os.path.join(data_dir, "simulated_example_mri")          # Replace with path for MRI data
taupet_root  = os.path.join(data_dir, "simulated_example_taupet")       # Replace with path for tau-PET data
mask_root    = os.path.join(data_dir, "simulated_example_fs")           # Replace with path for freesurfer mask data

# Config files
regions_csv  = os.path.join(repo_root, "src", "utils", "cho_stages.csv")

# Checkpoints and outputs
weights_path           = os.path.join(repo_root, "outputs", "ckpt", "unet_xxxxxx-xxxxx", "epochx.hdf5")  # Replace "x" with name of weights from U-Net epoch
eval_output_path       = os.path.join(repo_root, "outputs", "eval", "evaluation_results.csv")              # Select save name
prediction_output_dir  = os.path.join(repo_root, "outputs", "synthetic_test_scans")

# Create output directories
os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
os.makedirs(prediction_output_dir, exist_ok=True)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
batch_size = 1
encoder_filters = (2,2,2,2,512) #To not make net too large! Set to (32,64,128,256,512) in the article

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def get_dataframe():
    """
    Load and preprocess dataframe for evaluation or prediction.
    """
    #impute missing age and plasma p-tau217 to mean training set value
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    fillna_age = df_train['age'].mean()
    scaler_age = StandardScaler()
    scaler_age = scaler_age.fit(df_train[['age']])
    
    fillna_plasma = df_train['plasma_ptau217'].mean()
    scaler_plasma = StandardScaler()
    scaler_plasma = scaler_plasma.fit(df_train[['plasma_ptau217']])
    
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
    
    mri_test_paths, taupet_test_paths, mask_test_paths = ih.build_file_paths(
    df_test, mri_root, taupet_root, mask_root
    )
    print("Test paths successfully read")

    plasma_values = df_test["plasma_ptau217_imputed"].values.astype(np.float32)
    age_values = df_test["age_imputed"].values.astype(np.float32)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (mri_test_paths, mask_test_paths, plasma_values, age_values)
    )

    test_dataset = test_dataset.map(
        lambda x1, x2, x3, x4: ih.load_multiple_modalities_test(x1, x2, x3, x4),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Test dataset successfully created")
    return test_dataset, taupet_test_paths


def save_and_process_batch(i, mri_test, image_mask, taupet_test_paths, suvr_regions, plasma_value, age_value, model):
    """
    Process one batch for evaluation.
    """
    print("starting batch " + str(i))

    path_val = taupet_test_paths[i]
    image_true = nib.load(path_val).get_fdata()

    prediction = model([mri_test, plasma_value, age_value], training=False)
    image_pred = prediction[0, :, :, :, 0]
    
    nifti_image = nib.Nifti1Image(image_pred.numpy(), np.eye(4))
    subject_name = path_val.split("/")[-2]
    save_path = os.path.join(prediction_output_dir, subject_name + ".nii.gz")
    nib.save(nifti_image, save_path)

    diff_image = np.abs(image_true - image_pred)

    ssim_value = ssim(
        image_true,
        image_pred.numpy(),
        data_range=tf.reduce_max(image_pred).numpy() - tf.reduce_min(image_pred).numpy(),
    )
    mae_value = np.mean(diff_image)

    nonzero_mask = (image_true != 0)
    nonzero_image = diff_image[nonzero_mask]
    mae_brain_value = np.mean(nonzero_image)

    real_sum_values, gen_sum_values, nr_vox = [], [], []
    for nr in suvr_regions["number"]:
        mask = image_mask[0, :, :, :, 0] == nr
        nr_vox.append(np.sum(mask))
        real_sum_values.append(np.sum(image_true[mask]))
        gen_sum_values.append(np.sum(image_pred[mask]))

    region_df = pd.DataFrame([nr_vox, real_sum_values, gen_sum_values]).T
    region_df.columns = [f"nr_voxels_idx_{i}", f"real_values_idx_{i}", f"gen_values_idx_{i}"]

    print("done with batch " + str(i))
    return mae_value, mae_brain_value, ssim_value, region_df

# --------------------------------------------------
# MAIN
# --------------------------------------------------
df_test = get_dataframe()
test_dataset, taupet_test_paths = build_test_dataset(df_test)

suvr_regions = pd.read_csv(regions_csv)

# Build model and load weights
model = build_tau_pet_unet(encoder_filters=encoder_filters,verbose=True)
model.load_weights(weights_path)

# Convert dataset to list for easier parallel processing
test_data_list = list(test_dataset)
indices = list(range(len(test_data_list)))

mri_tests = [data[0] for data in test_data_list]
image_masks = [data[1] for data in test_data_list]
plasma_tests = [data[2] for data in test_data_list]
age_tests = [data[3] for data in test_data_list]


def process_batch_wrapper(i, mri_test, image_mask, taupet_test_paths, suvr_regions, plasma_value, age_value):
    return save_and_process_batch(i, mri_test, image_mask, taupet_test_paths, suvr_regions, plasma_value, age_value, model)

with ThreadPoolExecutor(max_workers=1) as executor:
    results = executor.map(
        process_batch_wrapper,
        indices,
        mri_tests,
        image_masks,
        [taupet_test_paths] * len(indices),
        [suvr_regions] * len(indices),
        plasma_tests,
        age_tests,
    )

mae, mae_brain, ssim_value, region_dfs = zip(*results)

# Combine region dataframes
suvr_regions_df = pd.concat([suvr_regions] + list(region_dfs), axis=1)

# Create evaluation dataframe
df_eval = pd.DataFrame(
    {
        "file_names": df_test["file_names"],
        "mae_full": mae,
        "mae_brain": mae_brain,
        "ssim": ssim_value,
    }
)

print("Evaluation DataFrame successfully created")

for name in suvr_regions_df["name"].unique():
    s = suvr_regions_df[suvr_regions_df["name"] == name]
    s = s[s.columns[3:]].sum(axis=0)

    true_suvr = []
    pred_suvr = []

    j = 0
    while j < len(s):
        nr_vox = s.iloc[j]
        j = j + 1
        true_suvr.append(s.iloc[j] / nr_vox)
        j = j + 1
        pred_suvr.append(s.iloc[j] / nr_vox)
        j = j + 1

    df_eval[name + "_true"] = true_suvr
    df_eval[name + "_pred"] = pred_suvr

df_eval.to_csv(eval_output_path, index=False)
print("Saved evaluation results to " + eval_output_path)

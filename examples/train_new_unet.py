"""
Training script for synthetic tau-PET prediction.
"""

# Imports
import os
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import StandardScaler

from src.models.unet3d import build_tau_pet_unet
from src.models.train import train
import src.utils.image_helpers as ih

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
batch_size = 4
epochs = 200
weight_sample = False
encoder_filters = (2,2,2,2,512) #To not make net too large! Set to (32,64,128,256,512) in the article

# Paths
train_csv = "datasets/simulated_example/simulated_example_df.csv" #Replace with path for train csv
val_csv = "datasets/simulated_example/simulated_example_df.csv" #Replace with path for val csv

mri_root = "datasets/simulated_example/simulated_example_mri" #Replace with path for MRI data
taupet_root = "datasets/simulated_example/simulated_example_taupet" #Replace with path for tau-PET data
mask_root = "datasets/simulated_example/simulated_example_fs" #Replace with path for MRI freesurfer mask data

# --------------------------------------------------
# SAVE NAME
# --------------------------------------------------
now = datetime.now().strftime("%y%m%d-%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "outputs")
log_dir = os.path.join(output_dir, "logs", f"unet_{now}")
ckpt_dir = os.path.join(output_dir, "ckpt", f"unet_{now}")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

summary_writer = tf.summary.create_file_writer(log_dir)
checkpoint_path = os.path.join(ckpt_dir, "")

# --------------------------------------------------
# RANDOM SEED
# --------------------------------------------------
np.random.seed(102)
tf.random.set_seed(102)
K.set_image_data_format("channels_last")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)

# Impute age
age_mean = df_train["age"].mean()
df_train["age_imputed"] = df_train["age"].fillna(age_mean)
df_val["age_imputed"] = df_val["age"].fillna(age_mean)

# Impute plasma p-tau217
plasma_mean = df_train["plasma_ptau217"].mean()
df_train["plasma_ptau217_imputed"] = df_train["plasma_ptau217"].fillna(plasma_mean)
df_val["plasma_ptau217_imputed"] = df_val["plasma_ptau217"].fillna(plasma_mean)

# Sample weights
if weight_sample:
    sample_weights = (df_train["suvr"].values ** 2).astype(np.float32)
else:
    sample_weights = np.ones(df_train.shape[0], dtype=np.float32)

# Build paths
mri_train_paths, taupet_train_paths, mask_train_paths = ih.build_file_paths(
    df_train, mri_root, taupet_root, mask_root
)
mri_val_paths, taupet_val_paths, mask_val_paths = ih.build_file_paths(
    df_val, mri_root, taupet_root, mask_root
)

print("Training paths successfully read")
print("Validation paths successfully read")

# --------------------------------------------------
# SCALE COVARIATES
# --------------------------------------------------
plasma_scaler = StandardScaler()
df_train["plasma_ptau217_imputed"] = plasma_scaler.fit_transform(
    df_train[["plasma_ptau217_imputed"]]
)
df_val["plasma_ptau217_imputed"] = plasma_scaler.transform(
    df_val[["plasma_ptau217_imputed"]]
)

age_scaler = StandardScaler()
df_train["age_imputed"] = age_scaler.fit_transform(df_train[["age_imputed"]])
df_val["age_imputed"] = age_scaler.transform(df_val[["age_imputed"]])

plasma_train_values = df_train["plasma_ptau217_imputed"].values.astype(np.float32)
plasma_val_values = df_val["plasma_ptau217_imputed"].values.astype(np.float32)

age_train_values = df_train["age_imputed"].values.astype(np.float32)
age_val_values = df_val["age_imputed"].values.astype(np.float32)

# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        mri_train_paths,
        taupet_train_paths,
        mask_train_paths,
        plasma_train_values,
        age_train_values,
        sample_weights,
    )
)

train_dataset = train_dataset.map(
    lambda x1, x2, x3, x4, x5, x6: ih.load_multiple_modalities_train(x1, x2, x3, x4, x5, x6),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("Training dataset successfully created")

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        mri_val_paths,
        taupet_val_paths,
        mask_val_paths,
        plasma_val_values,
        age_val_values,
    )
)

val_dataset = val_dataset.map(
    lambda x1, x2, x3, x4, x5: ih.load_multiple_modalities_train(x1, x2, x3, x4, x5),
    num_parallel_calls=tf.data.AUTOTUNE,
)

val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("Validation dataset successfully created")


# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------
model = build_tau_pet_unet(encoder_filters=encoder_filters,verbose=True)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-5,
    beta_1=0.7,
    beta_2=0.999,
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
train(
    model=model,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    summary_writer=summary_writer,
    checkpoint_path=checkpoint_path,
    epochs=epochs,
    alpha=0.5,
)

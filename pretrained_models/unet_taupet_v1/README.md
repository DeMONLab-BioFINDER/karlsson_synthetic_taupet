# Trained U-Net for tau-PET synthesis (v.1)

This dir contains the trained model in the article https://www.medrxiv.org/content/10.64898/2026.05.06.26352540v1. Training has been performed on 3815 individuals with MRI and age, for which 1708 also had plasma p-tau217 available. Individuals are from 13 different Alzheimer's disease cohorts representing a broad spectrum of clinical diagnoses. More details can be found in the research article.
The model architecture is specified in unet3d_v1.py and the corresponding weights in weights_v1.hdf5. For optimal performance, preprocess the MRIs according to the specifics in README on the first page/the research article.

## Inference
**Single new case**
For a single new case, a synthetic scan can be generated, visualized and saved as a NIfTI image using generate_single_taupet.ipynb.

**Multiple new cases***
For a test set with multiple cases, synthetic scans can be generated and saved as NIfTI images using generate_multiple_taupet.py.  

**Missing values**
Missing age and plasma p-tau217 values were imputed to the mean training set value for model flexiblity without adding new information. These values are provided as fillna_age and fillna_plasma. Note that generating synthetic scans on individuals that are missing these values will likely affect performance.

**Scalers**
All values need to be rescaled (z-scored) according to the training set transformation beforehand. The corresponding scalers are provided as scaler_age.joblib and scaler_plasma.joblib.

## Preprocessing

- T1w MRI (required): Skull stripping, registered to 2x2x2mm MNI space and saved as NIfTI.
- T1w MRI (optional, to evaluate performance): FreeSurfer v.6.0 segmentation based on the Desikan-Killiany atlas.
- Tau-PET (optional, to evaluate performance): attenuation-correction, motion-correction, summed, and rigid co-registration to MRI, smoothing to FWHM = 7 mm using an isotropic Gaussian kernel. Registered to 2x2x2mm MNI space and saved as NIfTI.
- Plasma p-tau217 (optional): measured with Eli Lilly immunoassays on a Meso Scale Discovery platform. Either use this assay or bridge the values for best performance. If plasma p-tau217 is not available, a synthetic scan will be generated without this information.

Note that voxel-wise standardization of MRI (z-scoring) within brain and background set to 0 is integrated within the training and evaluation functions of this repo, so not needed to do beforehand.

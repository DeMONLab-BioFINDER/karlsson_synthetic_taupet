[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) 
Author: Linda Karlsson, 2026. 

## Generating synthetic tau-PET scans in Alzheimer’s disease from MRI, blood biomarkers and demographics with deep learning

This repo contains code for training, evaluating and visualizing a U-Net 3D model that generates synthetic tau-PET scans from MRI, age and plasma p-tau217. It also contains a pre-trained version of the model that can be directly applied to new test cases, corresponding to the version described and evaluated in the article: https://www.medrxiv.org/content/10.64898/2026.05.06.26352540v1.

**Disclaimer:** This model is intended for research use only. Synthetic tau-PET images may contain errors and should not be used as a substitute for clinically acquired tau-PET or for clinical decision-making.

**Example of synthetic tau-PET scans generated on left-out test data**

https://github.com/user-attachments/assets/1b2dc483-1c58-4974-bdff-2b60bb8e8a18

## Background 
Tau protein aggregation in the brain is a hallmark of Alzheimer’s disease (AD). Tau-PET is the only method that can provide information regarding both the extent and distribution of tau pathology in vivo, increasing diagnostic confidence of physicians, capturing distinct disease heterogeneity related to clinical presentation and prognosis in AD, and helping differentiate AD from other neurodegenerative disorders. But despite its clear value, tau-PET remains largely unavailable to most patients and memory clinics worldwide due to its high cost and limited scalability. To address this, we developed a deep learning model trained to generate synthetic tau-PET scans from more accessible data: structural MRI, blood biomarkers and age.

We assembled data from 13 different AD cohorts, totaling 5,191 participants, and systematically optimized, tested and evaluated different deep learning models. The final model was a 3D image-to-image U-Net model with residual and attention units that integrated tabular variables in the bottleneck, see figure. Synthetic scans generated on left-out test data showed promising resemblance to true tau-PET in quantitative, qualitative and prognostic evaluations, with more details in the research article. We hope this model can help generate rich and more informative data in scenarios where true tau-PET is not available. However, keep in mind that the synthetic scans can contain errors and we do not recommend using them to replace true tau-PET.

<p align="center">
<img width="600" alt="Unet" src="https://github.com/user-attachments/assets/be70465d-3f39-41a7-8b18-0e521d051bab" />
</p>

## Structure and usage

```
karlsson_synthetic_taupet/
├── datasets/                    # dir for training, validation and test data
│   └── simulated_examples/      # simulated dataset for reference and correct formatting
├── examples/                    # examples of how to train and evaluate a new model
├── notebooks/                   # notebooks to evaluate results and create plots similar to those in the article
├── outputs/                     # dir for outputs during training and evaluation
│   ├── ckpt/                    # saved checkpoints during training 
│   ├── eval/                    # saved csv summary files during evaluation
│   ├── figures/                 # saved figures
│   ├── logs/                    # saved logs during training
│   └── synthetic_test_scans/    # saved generated synthetic tau-PET scans during evaluation
├── pretrained_models/           # dir for saved models
│   └── unet_taupet_v1/          # fitted synthetic tau-PET model
└── src/                         # source code
    ├── models/                  # code for creating and training new models
    └── utils/                   # code for help functions, pre-trained scalers, and composite definitions
```

Note that this repo contains both 1) code for replicating the work in the research article by training a new model and evaluating it, as well as 2) a pretrained model that can be used to generate synthetic tau-PET from T1w MRI, age and plasma p-tau217 for new subjects.

## Inference
New synthetic tau-PET scans can be generated from MRI, age and plasma p-tau217 using the pretrained model in pretrained_models/unet_taupet_v1/. The directory contains a notebook to generate, visualize and save a synthetic scan for a single new case (generate_single_taupet.ipynb), and a python script to generate and save synthetic scans for a dataset with multiple new cases (generate_multiple_taupet.py). See more details in pretrained_models/unet_taupet_v1/README.md.

## Data preprocessing

**Imaging**
In the research article, the following preprocessing was performed before training:
- ***T1w MRI:*** Skull stripping, FreeSurfer v.6.0 segmentation based on the Desikan-Killiany atlas.
- ***Tau-PET:*** attenuation-correction, motion-correction, summed, and rigid co-registration to MRI, smoothing to FWHM = 7 mm using an isotropic Gaussian kernel.
- ***All images:*** registered to 2x2x2mm MNI space and saved as NIfTI.

Note that voxel-wise standardization of MRI (z-scoring) within brain and background set to 0 is integrated within the training and evaluation functions of this repo, so not needed to do beforehand.

**Blood biomarker**
For the blood biomarker plasma p-tau217, it was measured with Eli Lilly immunoassays on a Meso Scale Discovery platform.

## Installation

1. This repository uses Git LFS to store the pre-trained model. Make sure to install Git LFS before cloning by running 
```bash
git lfs install
```

2. Clone the repository and navigate into it:
```bash
git clone https://github.com/DeMONLab-BioFINDER/karlsson_synthetic_taupet.git
cd karlsson_synthetic_taupet
```

3. [Option 1: Conda] Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate dl-env
```

***ONLY macOS (Apple Silicon M1/M2/M3)***
TensorFlow must be installed separately after the conda environment is set up:
```bash
pip install tensorflow-macos==2.9.0
pip install tensorflow-metal   # optional, for GPU support
```

3. [Option 2: venv] Create and activate the venv.
```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Install the packages in editable mode:
```bash
pip install -e .
```

## Dependencies

- python=3.9
- pip=23.3.1
- tensorflow==2.8.0
- numpy==1.22.4
- pandas==1.5.3
- scikit-learn==1.1.3
- nibabel==5.1.0
- tqdm==4.66.5
- jupyter==1.0.0
- seaborn==0.12.2
- matplotlib==3.7.3
- scikit-image==0.19.3
- protobuf==3.20.3

To set-up a virtual environment with the correct dependencies, run:
```
conda env create -f environment.yml
```

## Contact

For any questions or inquiries, please contact linda.karlsson@med.lu.se



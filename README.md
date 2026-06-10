Author: Linda Karlsson, 2026.

## Generating synthetic tau-PET scans in Alzheimer’s disease from MRI, blood biomarkers and demographics with deep learning

This repo contains code for training, evaluating and visualizing a U-Net 3D model that generates synthetic tau-PET scans from MRI, age and plasma p-tau217. It also contains a pre-trained version of the model that can be directly applied to new test cases, corresponding to the version evaluated in the article: https://www.medrxiv.org/content/10.64898/2026.05.06.26352540v1.

**Example of synthetic tau-PET scans generated on left-out test data**

https://github.com/user-attachments/assets/1b2dc483-1c58-4974-bdff-2b60bb8e8a18

## Background 
Tau protein aggregation in the brain is a hallmark of Alzheimer’s disease (AD). Tau-PET is the only method that can provide information regarding both the extent and distribution of tau pathology in vivo, increasing diagnostic confidence of physicians, capturing distinct disease heterogeneity related to clinical presentation and prognosis in AD, and helping differentiate AD from other neurodegenerative disorders. But despite its clear value, tau-PET remains largely unavailable to most patients and memory clinics worldwide due to its high cost and limited scalability. To address this, we developed a deep learning model trained to generate synthetic tau-PET scans from more accessible data: structural MRI, blood biomarkers and age.

We assembled data from 13 different AD cohorts, totaling 5,191 participants, and systematically tested and evaluated different deep learning models. The final model was a 3D image-to-image U-Net model with residual and attention units that integrated tabular variables in the bottleneck, see figure. Synthetic scans generated on left-out test data showed promising resemblance to true tau-PET in quantitative, qualitative and prognostic evaluations, with more details in the research article.

<img width="984" height="960" alt="Unet" src="https://github.com/user-attachments/assets/be70465d-3f39-41a7-8b18-0e521d051bab" />

## Structure and usage

```
karlsson_synthetic_taupet/
├── datasets/                    # dir for training, validation and test data
│   └── simulated_examples/      # simulated dataset for reference and correct formatting
├── examples/                    # examples of how to train and evaluate a new model
├── notebooks                    # notebooks to evaluate results and create plots similar to those in the article
├── pretrained_models            # dir for saved models
│   └── unet_taupet_v1/          # fitted synthetic tau-PET model
└── src                          # source code
    ├── models/                  # code for creating and training new models
    └── utils/                   # code for help functions, pre-trained scalers, and composite definitions
```

Note that this repo contains both 1) code for replicating the work in the research article by training a new model and evaluating it, as well as 2) a pretrained model that can be used to generate synthetic tau-PET from T1w MRI, age and plasma p-tau217 for new subjects.

## Preprocessing

In the research article, the following preprocessing was performed before training:
- ***T1w MRI:*** Skull stripping, FreeSurfer v.6.0 segmentation based on the Desikan-Killiany atlas.
- ***Tau-PET:*** attenuation-correction, motion-correction, summed, and rigid co-registration to MRI, smoothing to FWHM = 7 mm using an isotropic Gaussian kernel.
- ***All images:*** registered to 2x2x2mm MNI space.

Note that voxel-wise standardization of MRI (z-scoring) within brain and background set to 0 is performed during training.

## Dependencies

Python, tensorflow based code. Further dependencies are specified in the environment.yml file that can be used during set-up.





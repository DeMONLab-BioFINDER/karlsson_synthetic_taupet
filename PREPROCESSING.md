# MRI & Tau-PET Pre-processing 

**Original methods and implementation were created by Olof Strandberg.**

This document specifies the imaging preprocessing pipeline that was applied before training and
evaluating the U-Net in Karlsson et al. 2026. The pre-trained models in this git repo are based on images with this kind of pre-processing. Performance may vary if other pre-processing methods are used.

---

## 1. Raw MRI T1 (and if available, Tau-PET) recommendations

### 1.1 MRI T1

The most important first step is a **high-quality brain extraction**. In the original work, we used Freesurfer (FS). All downstream steps utilized two of the FS outputs (in NIfTI format).

1. The brain-extracted T1
2. Brain segmentation and parcellation based on the Desikan-Killiany atlas (`aparc+aseg.nii.gz`). Used in downstream evaluations and in the `[aa]` step to refine the brain mask.

### 1.2 Tau-PET (if available)

- Recon, attenuation correction, motion correction, averaging.
- Rigid co-registration of PET to MRI, after the MRI brain extraction.
- SUVR calculation using the **inferior cerebellum** as reference region.
- Smoothing to FWHM = 7 mm using an isotropic Gaussian kernel.

---

## 2. Pipeline structure

Below are the specific streams for each imaging modality after step 1. For MRI T1, the voxels inside the brain are firsly z-scored `[ss]`. Brain masking is then refined for MRI T1 and Tau-PET using the FS mask `[aa]`. Thereafter, all modalities are rigidly co-registered to a common MNI152 2×2×2mm space `[rr]` and cropped to **72×90×76** `[cr]`. The rigid transform is estimated **once**, from the T1 (step 6), and reused for the tau-PET and FS mask (step 7).

```
MRI T1     : brain ──────[ss]──> ──[aa]──> ──[rr]──> ──[cr]──> brain_ss_aa_rr_cr
FS mask    : aparc+aseg ─────────────────────[rr]──> ──[cr]──> mask_rr_cr
Tau-PET    : suvr ─────────────────[aa]──────[rr]──> ──[cr]──> suvr_aa_rr_cr
```
---

## 3. Standard scaling of MRI (`ss`)

> **Note.** Standard scaling is also built into the synthetic tau-PET model, so
> this step is probably not necessary, but was performed in the original work.

**Input:** Brain-extracted T1. **Note: never apply this step to PET.**

**Output:** `ssd.nii.gz`

**Algorithm:** z-score over non-zero voxels only; background stays exactly zero.

$$x' = \frac{x - \mu}{\sigma}$$

where $$\mu$$ and $$\sigma$$ are computed over non-zero voxels only.

```python
data = nib.load(f).get_fdata()
mask = data != 0
mean = np.mean(data[mask])
std  = np.std(data[mask])
data[mask] = (data[mask] - mean) / std
nib.Nifti1Image(data, affine).to_filename("ssd.nii.gz")
```

---

## 4. Brain masking from FreeSurfer mask (`aa`, part 1)
The `aa` step provides a marginal refinement of the
brain extraction that better delineates grey matter. It mainly affects
regions just outside the cortex, so the impact is expected to be small when
the initial brain extraction is already good.

**Input:** `aparc+aseg.nii.gz`

**Output:** `aabrainmask_d4e3.nii.gz`

**Algorithm:** binarise → dilate (4 iterations) → fill holes → erode (3 iterations).

```python
im = nib.load(f)
data = im.get_fdata()

mask = ndimage.binary_erosion(
    ndimage.binary_fill_holes(
        ndimage.binary_dilation(data, iterations=4)
    ),
    iterations=3,
)
nib.Nifti1Image(mask.astype("float64"), im.affine).to_filename("aabrainmask_d4e3.nii.gz")
```

---

## 5. Apply the brain mask to MRI and tau-PET (`aa`, part 2)

**Input:** the image to mask (T1 `ssd.nii.gz`, or tau-PET `suvr.nii.gz`) and the
matching mask `aabrainmask_d4e3.nii.gz` for that subject (from step 4).

**Output:** `image_aa.nii.gz`

**Algorithm:** zero every voxel outside the mask.

```python
mask_data = nib.load(mask_path).get_fdata()
img       = nib.load(image_path)
data      = img.get_fdata()
data[mask_data == 0] = 0
nib.Nifti1Image(data, img.affine).to_filename("image_aa.nii.gz")
```

---

## 6. Rigid registration to MNI with MRI (`rr`, part 1)

**Input:** `brain_ss_aa.nii.gz` (T1 MRI from step 5)

**Reference:** FSL `$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz`

**Outputs**, in the subject directory:

| File | Content |
|---|---|
| `transform_rigid0DerivedInitialMovingTranslation.mat` | initial translation stage |
| `transform_rigid1Rigid.mat` | rigid (6-DOF) stage |
| `transform_Warped_rigid.nii.gz` | the registered brain |

**Command** (absolute paths shortened for readability):

```bash
antsRegistration --float \
  --collapse-output-transforms 0 \
  --dimensionality 3 \
  --initialize-transforms-per-stage 0 \
  --interpolation Linear \
  --output [ ./transform_rigid, ./transform_Warped_rigid.nii.gz ] \
  --transform Rigid[ 0.1 ] \
  --metric MI[ MNI152_T1_2mm_brain.nii.gz, ./brain_ss_aa.nii.gz, 1, 32, None, 1 ] \
  --convergence [ 100x100x50, 1e-06, 10 ] \
  --smoothing-sigmas 9.0x6.0x3.0vox \
  --shrink-factors 8x4x2 \
  --use-histogram-matching 1 \
  --winsorize-image-intensities [ 0.0001, 0.9999 ] \
  --write-composite-transform 0 \
  --initial-moving-transform [ MNI152_T1_2mm_brain.nii.gz, ./brain_ss_aa.nii.gz, 1 ]
```

---

## 7. Apply the transforms from step 6 to FS mask and tau-PET (`rr`, part 2)

Used for the FS mask and tau-PET, so both are registered to the same space as the T1.

**Inputs**: `aparc+aseg.nii.gz` (FS mask), `suvr_aa.nii.gz` (tau-PET from step 5)

**Outputs**: `image_rr`

**Command** (absolute paths shortened for readability):

```bash
antsApplyTransforms -d 3 \
  -i aparc+aseg.nii.gz \
  -r MNI152_T1_2mm_brain.nii.gz \
  -o image_rr.nii.gz \
  --interpolation NearestNeighbor \
  -t transform_rigid1Rigid.mat \
  -t transform_rigid0DerivedInitialMovingTranslation.mat
```

---

## 8. Crop all images to the template bounding box (`cr`)

**Input:** `transform_Warped_rigid.nii.gz` (MRI brain), `mask_rr.nii.gz` (FS mask) 
or `suvr_aa_rr.nii.gz` (Tau-PET)

**Template:** the same `MNI152_T1_2mm_brain.nii.gz` used for registration.

**Output:** cropped input image to **72×90×76** (correct input dimensions 
for synthetic tau-PET model)

**Algorithm:** find each axis's bounding box from the template's non-zero
extent, then slice every input volume with those bounds.

```python
def getlim(d, t, oflo=0, ofhi=0):
    shape = d.shape
    lower = upper = 0
    for i in range(shape[t]):
        if i > 0:
            if   t == 0: prev, curr = np.sum(d[i-1, :, :]), np.sum(d[i, :, :])
            elif t == 1: prev, curr = np.sum(d[:, i-1, :]), np.sum(d[:, i, :])
            else:        prev, curr = np.sum(d[:, :, i-1]), np.sum(d[:, :, i])
            if prev == 0 and curr > 0:   lower = i   # zero -> non-zero
            elif prev > 0 and curr == 0: upper = i   # non-zero -> zero
    return lower + oflo, upper - ofhi

bounds = [getlim(template_data, ax) for ax in (0, 1, 2)]
(x0, x1), (y0, y1), (z0, z1) = bounds
cropped = img.get_fdata()[x0:x1, y0:y1, z0:z1]
nib.Nifti1Image(cropped, img.affine).to_filename(out)
```


## 10. Verification

After the final crop, check that everything looks correct by overlaying the output images. 
The cropped FS mask should sit directly on top of the cropped brain:

- `mask_rr_cr` overlays `brain_ss_aa_rr_cr` (and `suvr_aa_rr_cr`)
- all volumes have shape **72×90×76**

---

## 11. Dependencies

- **Python 3** with `nibabel`, `numpy`, `scipy`
- **ANTs** — `antsRegistration`, `antsApplyTransforms`
- **FSL** — the MNI152 2×2×2mm brain template
  (`$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz`)
- **FreeSurfer** — `recon-all`, run beforehand (section 1.1)

---

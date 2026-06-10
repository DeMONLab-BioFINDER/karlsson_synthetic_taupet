"""
Visualization utilities for synthetic tau-PET training.
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import ndimage


def generate_comparison_figure(
    prediction,
    mri,
    target,
    slice_indices=(10, 20, 30, 40, 50),
    mri_vmin=-1.5,
    mri_vmax=1.5,
    pet_vmin=0.0,
    pet_vmax=2.0,
):
    """
    Create a figure comparing MRI, synthetic tau-PET, and true tau-PET across multiple slices.

    Displays axial, coronal, and sagittal views for each specified slice index.
    The first column shows MRI, the second shows the model prediction (synthetic tau-PET),
    and the third shows the ground truth (true tau-PET).

    Parameters
    ----------
    prediction : tf.Tensor or np.ndarray, shape (batch, H, W, D, 1), dtype float32
        Network predictions (synthetic tau-PET images).
    mri : tf.Tensor or np.ndarray, shape (batch, H, W, D, 1), dtype float32
        Input MRI images.
    target : tf.Tensor or np.ndarray, shape (batch, H, W, D, 1), dtype float32
        Ground truth tau-PET images.
    slice_indices : tuple of int, optional
        Z-axis indices (depth dimension) for slices to display. Default is (10, 20, 30, 40, 50).
    mri_vmin : float, optional
        Minimum intensity for MRI visualization. Default is -1.5.
    mri_vmax : float, optional
        Maximum intensity for MRI visualization. Default is 1.5.
    pet_vmin : float, optional
        Minimum intensity for PET visualization. Default is 0.0.
    pet_vmax : float, optional
        Maximum intensity for PET visualization. Default is 2.0.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure object containing the comparison grid (rows x 9 columns).
    """
    prediction = np.asarray(prediction)
    mri = np.asarray(mri)
    target = np.asarray(target)

    n_rows = len(slice_indices)
    figure, axes = plt.subplots(n_rows, 9, figsize=(30, 4 * n_rows), constrained_layout=True)

    for row, slice_idx in enumerate(slice_indices):
        axes[row, 0].imshow(mri[0, :, :, slice_idx, 0], cmap="gray", vmin=mri_vmin, vmax=mri_vmax)
        axes[row, 1].imshow(prediction[0, :, :, slice_idx, 0], vmin=pet_vmin, vmax=pet_vmax)
        axes[row, 2].imshow(target[0, :, :, slice_idx, 0], vmin=pet_vmin, vmax=pet_vmax)

        axes[row, 3].imshow(ndimage.rotate(mri[0, :, slice_idx, :, 0], 90), cmap="gray", vmin=mri_vmin, vmax=mri_vmax)
        axes[row, 4].imshow(ndimage.rotate(prediction[0, :, slice_idx, :, 0], 90), vmin=pet_vmin, vmax=pet_vmax)
        axes[row, 5].imshow(ndimage.rotate(target[0, :, slice_idx, :, 0], 90), vmin=pet_vmin, vmax=pet_vmax)

        axes[row, 6].imshow(mri[0, slice_idx, :, :, 0], cmap="gray", vmin=mri_vmin, vmax=mri_vmax)
        axes[row, 7].imshow(prediction[0, slice_idx, :, :, 0], vmin=pet_vmin, vmax=pet_vmax)
        axes[row, 8].imshow(target[0, slice_idx, :, :, 0], vmin=pet_vmin, vmax=pet_vmax)

        for col in range(9):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    figure.suptitle("Axial, coronal, and sagittal views", fontsize=16)
    return figure


def figure_to_tensor(figure):
    """
    Convert a Matplotlib figure to a TensorFlow image tensor.

    Saves the figure as a PNG to a byte buffer, decodes it as a TensorFlow
    image tensor, and adds a batch dimension.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Matplotlib figure object to convert.

    Returns
    -------
    image : tf.Tensor, shape (1, H, W, 4), dtype uint8
        Image tensor with batch dimension and RGBA channels.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)

    buffer.seek(0)
    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, axis=0)
    return image

"""
Loss function for synthetic tau-PET training.
"""

import tensorflow as tf

def masked_l1_loss(prediction, target, mask=1.0, alpha=0.5, sample_weight=1.0):
    """
    Compute a weighted combination of whole-image and masked L1 loss.

    Calculates two L1 loss components: whole-image L1 loss over all voxels 
    and masked L1 loss over voxels where the mask is non-zero. The final loss 
    is: whole_image_loss + alpha * masked_loss.

    Parameters
    ----------
    prediction : tf.Tensor, shape (batch, H, W, D, 1), dtype float32
        Network predictions (synthetic tau-PET images).
    target : tf.Tensor, shape (batch, H, W, D, 1), dtype float32
        Ground truth tau-PET images.
    mask : tf.Tensor or float, shape (batch, H, W, D, 1) or scalar, dtype float32, optional
        Binary mask indicating region of interest (e.g., brain mask). 
        Default is 1.0 (no masking).
    alpha : float, optional
        Weighting factor for the masked loss component. Default is 0.5.
    sample_weight : tf.Tensor or float, shape (batch,) or scalar, dtype float32, optional
        Per-sample weights applied to the masked loss. Default is 1.0.

    Returns
    -------
    loss : tf.Tensor, shape (), dtype float32
        Scalar tensor containing the combined loss value.
    """
    absolute_error = tf.abs(target - prediction)
    whole_image_loss = tf.reduce_mean(absolute_error)
    masked_loss = tf.reduce_mean(sample_weight * tf.multiply(tf.abs(target - prediction), mask))

    return whole_image_loss + alpha * masked_loss

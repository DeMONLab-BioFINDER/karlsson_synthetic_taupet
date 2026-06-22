"""
Training utilities for synthetic tau-PET generation.
"""

import tensorflow as tf
from tqdm.auto import tqdm

from src.models.loss import masked_l1_loss
from src.models.training_visualization import figure_to_tensor, generate_comparison_figure


@tf.function
def train_step(
    model,
    optimizer,
    mri,
    target,
    mask=1.0,
    plasma=0.0,
    age=0.0,
    sample_weight=1.0,
    alpha=0.5,
):
    """
    Run one forward pass, compute the loss, and apply gradients.

    Computes the masked L1 loss between the model prediction and the target
    inside a GradientTape, then updates model weights via the optimizer.

    Parameters
    ----------
    model : keras.Model
        The 3D U-Net model to train.
    optimizer : keras.optimizers.Optimizer
        Optimizer used to apply gradients.
    mri : tf.Tensor, shape (batch, H, W, D, 1), dtype float32
        Input MRI images.
    target : tf.Tensor, shape (batch, H, W, D, 1), dtype float32
        Ground truth tau-PET images.
    mask : tf.Tensor or float, shape (batch, H, W, D, 1) or scalar, dtype float32, optional
        Brain mask applied to the masked loss component. Default is 1.0.
    plasma : tf.Tensor or float, shape (batch, 1) or scalar, dtype float32, optional
        Plasma biomarker covariate. Default is 0.0.
    age : tf.Tensor or float, shape (batch, 1) or scalar, dtype float32, optional
        Age covariate. Default is 0.0.
    sample_weight : tf.Tensor or float, shape (batch,) or scalar, dtype float32, optional
        Per-sample weights for the masked loss. Default is 1.0.
    alpha : float, optional
        Weighting factor for the masked loss component. Default is 0.5.

    Returns
    -------
    loss : tf.Tensor, shape (), dtype float32
        Scalar loss value for this training step.
    """
    with tf.GradientTape() as tape:
        prediction = model([mri, plasma, age], training=True)
        loss = masked_l1_loss(
            prediction=prediction,
            target=target,
            mask=mask,
            alpha=alpha,
            sample_weight=sample_weight,
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def validation_step(
    model,
    mri,
    target,
    mask=1.0,
    plasma=0.0,
    age=0.0,
    alpha=0.5,
):
    """
    Run one forward pass in inference mode and compute the validation loss.

    Parameters
    ----------
    model : keras.Model
        The 3D U-Net model to evaluate.
    mri : tf.Tensor, shape (batch, H, W, D, 1), dtype float32
        Input MRI images.
    target : tf.Tensor, shape (batch, H, W, D, 1), dtype float32
        Ground truth tau-PET images.
    mask : tf.Tensor or float, shape (batch, H, W, D, 1) or scalar, dtype float32, optional
        Brain mask applied to the masked loss component. Default is 1.0.
    plasma : tf.Tensor or float, shape (batch, 1) or scalar, dtype float32, optional
        Plasma biomarker covariate. Default is 0.0.
    age : tf.Tensor or float, shape (batch, 1) or scalar, dtype float32, optional
        Age covariate. Default is 0.0.
    alpha : float, optional
        Weighting factor for the masked loss component. Default is 0.5.

    Returns
    -------
    loss : tf.Tensor, shape (), dtype float32
        Scalar validation loss value for this batch.
        """
    prediction = model([mri, plasma, age], training=False)
    loss = masked_l1_loss(
        prediction=prediction,
        target=target,
        mask=mask,
        alpha=alpha,
    )
    return loss


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    summary_writer=None,
    checkpoint_path=None,
    epochs=1,
    alpha=0.5,
):
    """
    Run the full training loop over a given number of epochs.

    At each epoch, iterates over the training dataset calling train_step,
    then evaluates on the validation dataset calling validation_step. Logs
    train/val losses and a prediction preview image to TensorBoard if a
    summary writer is provided. Saves model weights at the best validation
    loss and every 10 epochs if a checkpoint path is provided.

    Parameters
    ----------
    model : keras.Model
        The 3D U-Net model to train.
    optimizer : keras.optimizers.Optimizer
        Optimizer used to update model weights.
    train_dataset : tf.data.Dataset
        Training dataset yielding batches of (mri, target, mask, plasma, age, ...).
    val_dataset : tf.data.Dataset
        Validation dataset yielding batches of (mri, target, mask, plasma, age, ...).
    summary_writer : tf.summary.SummaryWriter or None, optional
        TensorBoard summary writer for logging losses and images. Default is None.
    checkpoint_path : str or None, optional
        Directory path prefix for saving model weight checkpoints (.hdf5).
        Default is None (no checkpointing).
    epochs : int, optional
        Number of training epochs. Default is 1.
    alpha : float, optional
        Weighting factor for the masked loss component passed to train_step
        and validation_step. Default is 0.5.

    """
    best_val_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Training"):
        train_losses = []

        for batch in tqdm(train_dataset, desc=f"Epoch {epoch + 1}", leave=False):
            mri, target, mask, plasma, age = batch[:5]
            loss = train_step(
                model=model,
                optimizer=optimizer,
                mri=mri,
                target=target,
                mask=mask,
                plasma=plasma,
                age=age,
                alpha=alpha,
            )
            train_losses.append(loss)

        val_losses = []
        preview_image = None

        for batch in val_dataset:
            mri, target, mask, plasma, age = batch[:5]
            prediction = model([mri, plasma, age], training=False)

            val_loss = validation_step(
                model=model,
                mri=mri,
                target=target,
                mask=mask,
                plasma=plasma,
                age=age,
                alpha=alpha,
            )
            val_losses.append(val_loss)

            if preview_image is None:
                figure = generate_comparison_figure(prediction, mri, target)
                preview_image = figure_to_tensor(figure)

        mean_train_loss = tf.reduce_mean(train_losses)
        mean_val_loss = tf.reduce_mean(val_losses)

        if summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar("train_loss", mean_train_loss, step=epoch)
                tf.summary.scalar("val_loss", mean_val_loss, step=epoch)
                if preview_image is not None:
                    tf.summary.image("Comparison", preview_image, step=epoch)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            if checkpoint_path is not None:
                model.save_weights(checkpoint_path + 'epoch{0}.hdf5'.format(epoch))
                
        elif epoch % 10 == 0:
            # checkpoint.save(file_prefix = checkpoint_prefix) 
            model.save_weights(checkpoint_path + 'epoch{0}.hdf5'.format(epoch))

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {mean_train_loss.numpy():.5f} - "
            f"val_loss: {mean_val_loss.numpy():.5f}"
        )

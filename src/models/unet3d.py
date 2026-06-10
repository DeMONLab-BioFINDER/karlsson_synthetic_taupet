"""
3D U-Net architecture for synthetic tau-PET generation.
"""

import numpy as np
import tensorflow as tf

from keras import Model
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv3D,
    Conv3DTranspose,
    Cropping3D,
    Dense,
    Input,
    LeakyReLU,
    MaxPooling3D,
    Reshape,
    ZeroPadding3D,
    add,
    concatenate,
)


def conv_block(inputs, num_filters, kernel_size=(3, 3, 3), activation=None, padding="same"):
    """
    Apply 3D convolution, batch normalization, and activation.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor.
    num_filters : int
        Number of convolutional filters.
    kernel_size : tuple of int, optional
        Convolution kernel size. Default is (3, 3, 3).
    activation : keras.layers.Layer or None, optional
        Activation layer. Default is LeakyReLU().
    padding : str, optional
        Padding mode for convolution. Default is "same".

    Returns
    -------
    x : tf.Tensor
        Output tensor after convolution, batch norm, and activation.
    """
    if activation is None:
        activation = LeakyReLU()

    x = Conv3D(num_filters, kernel_size, padding=padding)(inputs)
    x = BatchNormalization(fused=False)(x)
    x = Activation(activation)(x)
    return x


def residual_block(inputs, num_filters):
    """
    Residual 3D convolutional block with skip connection.

    Two convolutions with batch normalization and LeakyReLU activations,
    with a skip connection adding the input to the output.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor with number of channels matching num_filters.
    num_filters : int
        Number of convolutional filters.

    Returns
    -------
    x : tf.Tensor
        Output tensor after residual connection.
    """
    x = Conv3D(num_filters, (3, 3, 3), padding="same")(inputs)
    x = BatchNormalization(fused=False)(x)
    x = Activation(LeakyReLU())(x)

    x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization(fused=False)(x)

    x = add([x, inputs])
    x = Activation(LeakyReLU())(x)
    return x


def attention_block(inputs):
    """
    Self-attention block for spatial features.

    Computes scaled dot-product attention over spatial voxels, designed for
    bottleneck feature maps with small spatial dimensions.

    Parameters
    ----------
    inputs : tf.Tensor
        Input feature map.

    Returns
    -------
    output : tf.Tensor
        Output tensor after attention mechanism and skip connection.
    """
    query = Conv3D(filters=inputs.shape[-1] // 8, kernel_size=1)(inputs)
    key = Conv3D(filters=inputs.shape[-1] // 8, kernel_size=1)(inputs)
    value = Conv3D(filters=inputs.shape[-1], kernel_size=1)(inputs)
    attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(inputs.shape[-1] // 8, dtype=tf.float32)))
    output = tf.matmul(attention_weights, value)
    output = add([output, inputs])

    return output


def down_block(inputs, num_filters):
    """
    Encoder block combining convolution and residual connection.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor.
    num_filters : int
        Number of convolutional filters.

    Returns
    -------
    x : tf.Tensor
        Output tensor after conv and residual blocks.
    """
    x = conv_block(inputs, num_filters)
    x = residual_block(x, num_filters)
    return x


def bottleneck_block(inputs, num_filters):
    """
    Bottleneck block with attention.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor.
    num_filters : int
        Number of convolutional filters.

    Returns
    -------
    x : tf.Tensor
        Output tensor after conv, attention, and conv blocks.
    """
    x = conv_block(inputs, num_filters)
    x = attention_block(x)
    x = conv_block(x, num_filters)
    return x


def up_block(inputs, skip_connection, num_filters):
    """
    Decoder block with transposed convolution and skip connection.

    Applies transposed convolution to upsample, concatenates with skip connection,
    and applies conv and residual blocks.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor (features from deeper layer).
    skip_connection : tf.Tensor
        Encoder feature map from corresponding down_block.
    num_filters : int
        Number of convolutional filters.

    Returns
    -------
    x : tf.Tensor
        Output tensor after upsampling, concatenation, and residual blocks.
    """
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding="same")(inputs)
    x = concatenate([x, skip_connection], axis=4)
    x = conv_block(x, num_filters)
    x = residual_block(x, num_filters)
    return x


def covariate_bottleneck_layer(covariate_input, spatial_shape=(4, 4, 4), hidden_units=1024, name="covariate"):
    """
    Project scalar covariates into a 3D spatial feature map.

    Passes the input through dense layers and reshapes the output to match
    the specified spatial dimensions for concatenation with CNN features.

    Parameters
    ----------
    covariate_input : tf.Tensor, shape (batch, 1)
        Scalar covariate value (e.g., plasma, age).
    spatial_shape : tuple of int, optional
        Spatial dimensions of the output feature map. Default is (4, 4, 4).
    hidden_units : int, optional
        Number of units in the hidden dense layer. Default is 1024.
    name : str, optional
        Name prefix for the layers. Default is "covariate".

    Returns
    -------
    x : tf.Tensor, shape (batch, *spatial_shape, 1)
        Reshaped feature map ready for concatenation.
    """
    num_voxels = int(np.prod(spatial_shape))
    
    if name == "plasma":
        dense1_name, dense2_name = "data1", "data2"
    elif name == "age":
        dense1_name, dense2_name = "data3", "data4"
    else:
        dense1_name, dense2_name = f"{name}_dense1", f"{name}_dense2"

    x = Dense(hidden_units, activation="relu", name=f"{name}_dense1")(covariate_input)
    x = Dense(num_voxels, activation="relu", name=f"{name}_dense2")(x)
    x = Reshape((*spatial_shape, 1), name=f"{name}_reshape")(x)
    return x


def build_tau_pet_unet(
    image_shape=(72, 90, 76, 1),
    padding=(28, 19, 26),
    encoder_filters=(32, 64, 128, 256, 512),
    bottleneck_filters=1024,
    covariate_shape=(4, 4, 4),
    output_activation=None,
    verbose=False,
):
    """
    Build a multimodal 3D U-Net for synthetic tau-PET generation.

    Constructs a U-Net encoder-decoder architecture with residual and attention
    blocks. Takes MRI and scalar covariates (plasma, age) as inputs and outputs
    synthetic tau-PET images. Covariates are projected into 3D bottleneck features
    and concatenated with CNN features.

    Parameters
    ----------
    image_shape : tuple of int, optional
        Input MRI image shape (H, W, D, channels). Default is (72, 90, 76, 1).
    padding : tuple of int, optional
        Zero padding applied to MRI input (H_pad, W_pad, D_pad). Default is (28, 19, 26).
    encoder_filters : tuple of int, optional
        Number of filters at each encoder level. Default is (32, 64, 128, 256, 512).
    bottleneck_filters : int, optional
        Number of filters in the bottleneck. Default is 1024.
    covariate_shape : tuple of int, optional
        Spatial shape of covariate bottleneck features. Default is (4, 4, 4).
    output_activation : keras.layers.Layer or None, optional
        Output activation layer. Default is LeakyReLU().
    verbose : bool, optional
        If True, print model summary. Default is False.

    Returns
    -------
    model : keras.Model
        Compiled U-Net model with inputs [mri, plasma, age] and output tau-PET.
    """
    if output_activation is None:
        output_activation = LeakyReLU()

    mri_input = Input(shape=image_shape, name="mri_input")
    plasma_input = Input(shape=(1,), name="plasma_input")
    age_input = Input(shape=(1,), name="age_input")

    x = ZeroPadding3D(padding=padding)(mri_input)

    skips = []
    for num_filters in encoder_filters:
        x = down_block(x, num_filters)
        skips.append(x)
        x = MaxPooling3D((2, 2, 2))(x)

    x = bottleneck_block(x, bottleneck_filters)

    plasma_features = covariate_bottleneck_layer(
        plasma_input,
        spatial_shape=covariate_shape,
        hidden_units=bottleneck_filters,
        name="plasma",
    )

    age_features = covariate_bottleneck_layer(
        age_input,
        spatial_shape=covariate_shape,
        hidden_units=bottleneck_filters,
        name="age",
    )

    x = concatenate([x, plasma_features], axis=4)
    x = concatenate([x, age_features], axis=4)

    for skip_connection, num_filters in zip(reversed(skips), reversed(encoder_filters)):
        x = up_block(x, skip_connection, num_filters)

    x = Conv3D(
        1,
        (1, 1, 1),
        activation=output_activation,
        bias_initializer="zeros",
    )(x)

    output = Cropping3D(cropping=padding)(x)

    model = Model(
        inputs=[mri_input, plasma_input, age_input],
        outputs=output,
        name="tau_pet_unet",
    )

    if verbose:
        model.summary()

    return model
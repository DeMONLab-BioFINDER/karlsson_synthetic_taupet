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
    Convolution -> batch normalization -> activation.
    """
    if activation is None:
        activation = LeakyReLU()

    x = Conv3D(num_filters, kernel_size, padding=padding)(inputs)
    x = BatchNormalization(fused=False)(x)
    x = Activation(activation)(x)
    return x


def residual_block(inputs, num_filters):
    """
    Residual 3D convolutional block.
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
    Self-attention block over spatial voxels.
    Intended for bottleneck feature maps where spatial dimensions are small.
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
    Encoder block.
    """
    x = conv_block(inputs, num_filters)
    x = residual_block(x, num_filters)
    return x


def bottleneck_block(inputs, num_filters):
    """
    Bottleneck block.
    """
    x = conv_block(inputs, num_filters)
    x = attention_block(x)
    x = conv_block(x, num_filters)
    return x


def up_block(inputs, skip_connection, num_filters):
    """
    Decoder block.
    """
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding="same")(inputs)
    x = concatenate([x, skip_connection], axis=4)
    x = conv_block(x, num_filters)
    x = residual_block(x, num_filters)
    return x


def covariate_bottleneck_layer(covariate_input, spatial_shape=(4, 4, 4), hidden_units=1024, name="covariate"):
    """
    Project scalar covariates into a 3D bottleneck feature map.
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
    Build the multimodal 3D U-Net.
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

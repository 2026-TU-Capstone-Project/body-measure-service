import tensorflow as tf


def get_side_encoder(batch_size=1, input_shape=(256, 256, 3), num_output_features=16):
    """
    측면 이미지(256x256x3) → 16-dim feature vector 경량 인코더.

    구조 (DepthwiseSeparableConv 스타일):
        256x256 → 128x128 (24ch)
        128x128 → 64x64  (32ch)
         64x64  → 32x32  (48ch)
         32x32  → 16x16  (64ch)
         16x16  → 8x8    (96ch)
        GlobalAvgPool → Dense(16)
    """
    inp = tf.keras.layers.Input(
        shape=(input_shape[0], input_shape[1], 3),
        batch_size=batch_size,
        name="side_image",
    )

    # Block 1: 256 → 128
    x = tf.keras.layers.Conv2D(
        24, kernel_size=3, strides=2, padding="same",
        activation="relu", name="se_conv1",
    )(inp)

    # Block 2: 128 → 64
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, padding="same", activation=None, name="se_dw2",
    )(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, activation="relu", name="se_pw2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="se_pool2")(x)

    # Block 3: 64 → 32
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, padding="same", activation=None, name="se_dw3",
    )(x)
    x = tf.keras.layers.Conv2D(48, kernel_size=1, activation="relu", name="se_pw3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="se_pool3")(x)

    # Block 4: 32 → 16
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, padding="same", activation=None, name="se_dw4",
    )(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, activation="relu", name="se_pw4")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="se_pool4")(x)

    # Block 5: 16 → 8
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, padding="same", activation=None, name="se_dw5",
    )(x)
    x = tf.keras.layers.Conv2D(96, kernel_size=1, activation="relu", name="se_pw5")(x)

    # GlobalAvgPool + FC
    x = tf.keras.layers.GlobalAveragePooling2D(name="se_gap")(x)
    x = tf.keras.layers.Dense(
        num_output_features, activation="relu", name="se_fc",
    )(x)

    return tf.keras.Model(inputs=inp, outputs=x, name="side_encoder")

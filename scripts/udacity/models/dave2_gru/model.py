from __future__ import annotations
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models, optimizers  # type: ignore

def build_dave2_backbone(image_shape: tuple[int, int, int]) -> tf.keras.Model:  # type: ignore
    """
    Dave2-like CNN backbone that maps a single frame to a feature vector.
    """
    img_in = layers.Input(shape=image_shape, name="img_in")
    x = layers.Lambda(lambda t: t / 255.0, name="norm")(img_in)

    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", name="conv2d_1")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu", name="conv2d_2")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), activation="relu", name="conv2d_3")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2d_4")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2d_5")(x)
    x = layers.Dropout(0.1)(x)

    feat = layers.Flatten(name="flattened")(x)
    return models.Model(inputs=img_in, outputs=feat, name="dave2_backbone")


def build_dave2_gru(input_shape: tuple[int, int, int], seq_len: int, num_outputs: int, learning_rate: float, gru_units: int = 128) -> tf.keras.Model:  # type: ignore
    """
    Sequence model: per-frame CNN backbone + GRU + linear head.

    Input:
        (T, H, W, C) with sequence length T = seq_len
    Output:
        num_outputs regression values per sequence.
    """
    h, w, c = input_shape

    cnn = build_dave2_backbone((h, w, c))

    seq_in = layers.Input(shape=(seq_len, h, w, c), name="seq_img_in")
    td_feat = layers.TimeDistributed(cnn, name="td_backbone")(seq_in)

    h_state = layers.GRU(gru_units, return_sequences=False, dropout=0.1, name="gru")(td_feat)
    h_state = layers.Dense(100, activation="relu", name="fc_100")(h_state)
    h_state = layers.Dense(50, activation="relu", name="fc_50")(h_state)

    out = layers.Dense(num_outputs, activation="linear", name="outputs")(h_state)

    model = models.Model(inputs=seq_in, outputs=out, name=f"dave2_gru_T{seq_len}")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model

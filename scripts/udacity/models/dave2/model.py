from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from .config import INPUT_SHAPE, NUM_OUTPUTS, LEARNING_RATE

def build_dave2(num_outputs: int = NUM_OUTPUTS) -> tf.keras.Model: # type: ignore
    """PilotNet/DAVE-2 backbone as in PD's example; no behavior changes."""
    row, col, ch = INPUT_SHAPE
    img_in = layers.Input(shape=(row, col, ch), name="img_in")
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
    x = layers.Flatten(name="flattened")(x)
    x = layers.Dense(100, activation="relu", name="fc100")(x)
    x = layers.Dense(50,  activation="relu", name="fc50")(x)
    out = layers.Dense(num_outputs, activation="linear", name="steering_throttle")(x)

    model = models.Model(inputs=img_in, outputs=out, name="dave2")
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]) # PD used "acc" in examples, but that's not meaningful for regression. So im using MAE.
    return model

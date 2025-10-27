from __future__ import annotations
from typing import Tuple
import sys
from pathlib import Path
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models, optimizers # type: ignore

# add project root to path
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer reusing your existing DAVE-2 config for shape & hyperparams.
try:
    from sims.udacity.models.dave2_gru.config import INPUT_SHAPE as _INPUT_SHAPE, LEARNING_RATE
    DEFAULT_LR = LEARNING_RATE
    ROW, COL, CH = _INPUT_SHAPE
except Exception:
    # Fallback — set to your PD Conf() values if needed.
    ROW, COL, CH = 120, 160, 3
    DEFAULT_LR = 1e-4


def build_dave2_backbone(image_shape: Tuple[int, int, int]) -> tf.keras.Model: # type: ignore
    """
    DAVE-2 / PilotNet-like CNN that outputs a flattened feature vector per frame.
    Matches the PD example structure (Conv blocks → Flatten).
    """
    img_in = layers.Input(shape=image_shape, name="img_in")
    x = layers.Lambda(lambda t: t / 255.0, name="norm")(img_in)

    # NVIDIA-style conv stack
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


def build_dave2_gru(seq_len: int,
                    num_outputs: int = 2,
                    learning_rate: float = DEFAULT_LR,
                    gru_units: int = 128) -> tf.keras.Model: # type: ignore
    """
    Wraps the DAVE-2 backbone in TimeDistributed, inserts a GRU between CNN and FC head.

    Input shape: (T, ROW, COL, CH)
    Output: linear regression head of size `num_outputs` (e.g., [steering, throttle])

    Notes:
    - With seq_len==1 the model reduces to a (nearly) frame-based setup while retaining GRU.
    - We keep the FC(100)->FC(50) head as in PD’s example and only route their input through GRU.
    """
    # Per-frame CNN
    cnn = build_dave2_backbone((ROW, COL, CH))

    # Sequence input
    seq_in = layers.Input(shape=(seq_len, ROW, COL, CH), name="seq_img_in")
    td_feat = layers.TimeDistributed(cnn, name="td_backbone")(seq_in)          # (B, T, F)

    # Temporal modeling
    h = layers.GRU(gru_units, return_sequences=False, dropout=0.1, name="gru")(td_feat)

    # Original FC head (moved after GRU)
    h = layers.Dense(100, activation="relu", name="fc_100")(h)
    h = layers.Dense(50, activation="relu", name="fc_50")(h)

    out = layers.Dense(num_outputs, activation="linear", name="steering_throttle")(h)

    model = models.Model(inputs=seq_in, outputs=out, name=f"dave2_gru_T{seq_len}")
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="acc")])
    return model

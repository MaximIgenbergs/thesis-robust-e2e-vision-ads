import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def _cnn_backbone(frame_shape):
    """PilotNet-style CNN applied to ONE frame."""
    inp = layers.Input(shape=frame_shape, name="frame_input")
    x = layers.Conv2D(24, 5, strides=2, activation='elu', name='conv1')(inp)
    x = layers.Conv2D(36, 5, strides=2, activation='elu', name='conv2')(x)
    x = layers.Conv2D(48, 5, strides=2, activation='elu', name='conv3')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='elu', name='conv4')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='elu', name='conv5')(x)
    x = layers.Flatten(name='flatten')(x)
    return Model(inp, x, name="cnn_backbone")

def build_model(frame_shape=(66, 200, 3), seq_len=5, gru_units=128, dropout=0.5):
    """
    TimeDistributed CNN -> GRU -> FC head.
    Input: (B, T, H, W, C)
    Output: (B, 2) => [steer, throttle]
    """
    seq_inp = layers.Input(shape=(seq_len, *frame_shape), name="image_sequence")

    cnn = _cnn_backbone(frame_shape)
    x = layers.TimeDistributed(cnn, name="td_cnn")(seq_inp)            # (B, T, F)
    x = layers.GRU(gru_units, return_sequences=False, name="gru")(x)   # (B, U)

    x = layers.Dense(1164, activation='elu', name='fc1')(x)
    x = layers.Dropout(dropout, name='dropout1')(x)
    x = layers.Dense(100, activation='elu', name='fc2')(x)
    x = layers.Dropout(dropout, name='dropout2')(x)
    x = layers.Dense(50, activation='elu', name='fc3')(x)
    x = layers.Dense(10, activation='elu', name='fc4')(x)

    steer    = layers.Dense(1, name='steering')(x)
    throttle = layers.Dense(1, name='throttle')(x)
    out      = layers.Concatenate(name='control_outputs')([steer, throttle])
    return Model(inputs=seq_inp, outputs=out, name='dave2_gru_v1')

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def build_model(input_shape=(66, 200, 3)):
    """
    Builds the baseline single-frame CNN model.
    Args:
        input_shape: tuple, (height, width, channels).
    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name='image_input')

    # Convolutional backbone
    x = layers.Conv2D(24, 5, strides=2, activation='elu', name='conv1')(inputs)
    x = layers.Conv2D(36, 5, strides=2, activation='elu', name='conv2')(x)
    x = layers.Conv2D(48, 5, strides=2, activation='elu', name='conv3')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='elu', name='conv4')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='elu', name='conv5')(x)

    x = layers.Flatten(name='flatten')(x)

    # Fully-connected head
    x = layers.Dense(1164, activation='elu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(100, activation='elu', name='fc2')(x)
    x = layers.Dropout(0.5, name='dropout2')(x)
    x = layers.Dense(50, activation='elu', name='fc3')(x)
    x = layers.Dense(10, activation='elu', name='fc4')(x)

    # Two-output regression
    steer    = layers.Dense(1, name='steering')(x)
    throttle = layers.Dense(1, name='throttle')(x)
    outputs  = layers.Concatenate(name='control_outputs')([steer, throttle])

    return Model(inputs=inputs, outputs=outputs, name='baseline_cnn')

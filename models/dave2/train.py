import sys
import pathlib

# Add project root to PYTHONPATH so shared utils can be imported
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import datetime
import tensorflow as tf
from matplotlib import pyplot as plt

# shared helpers and config
from models.utils.utils import load_dataframes
from models.utils.paths import get_model_dir, get_fig_dir, TRAIN_IMG_DIR, TRAIN_LOG_PATH
from models.utils.training_defaults import (
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    ALPHA_STEER,
    VAL_SPLIT,
    RANDOM_SEED,
)

# model-specific imports
from models.dave2.utils.data_loader import DrivingDataset
from models.dave2.model import build_model

# paths
LOG_PATH = TRAIN_LOG_PATH
IMG_DIR = TRAIN_IMG_DIR
MODELS_DIR = get_model_dir('dave2')
FIG_DIR = get_fig_dir('dave2')

# load & split CSV 
train_df, val_df = load_dataframes(
    LOG_PATH,
    val_split=VAL_SPLIT,
    random_seed=RANDOM_SEED
)

# build tf.data pipelines 
train_ds = DrivingDataset(
    df=train_df,
    img_dir=IMG_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True
).dataset()

val_ds = DrivingDataset(
    df=val_df,
    img_dir=IMG_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False
).dataset()

# instantiate model 
model = build_model(input_shape=(66, 200, 3))

# weighted MSE loss 
def weighted_mse(y_true, y_pred):
    steer_true, throttle_true = y_true[:, 0], y_true[:, 1]
    steer_pred, throttle_pred = y_pred[:, 0], y_pred[:, 1]
    loss_s = tf.reduce_mean(tf.square(steer_true - steer_pred))
    loss_t = tf.reduce_mean(tf.square(throttle_true - throttle_pred))
    return ALPHA_STEER * loss_s + (1 - ALPHA_STEER) * loss_t

# compile 
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=weighted_mse,
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]
)

# callbacks 
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
]

# train 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# save final model 
model.save(str(MODELS_DIR / 'final_model.h5'))
print(f"Training complete. Model saved to: {MODELS_DIR/'final_model.h5'}")

# plot & save loss curve 
plt.figure()
plt.plot(history.history['loss'],   label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

ts = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
plot_name = f"loss_curve_{ts}.png"
plt.savefig(str(FIG_DIR / plot_name))
plt.close()
print(f"Loss curve saved to: {FIG_DIR/plot_name}")

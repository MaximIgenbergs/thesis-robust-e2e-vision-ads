import os
import tensorflow as tf
from utils.data_loader import DrivingDataset
from utils.utils import load_dataframes
from model import build_model

# Configuration
CSV_PATH      = 'udacity_dataset_lake_dave/jungle_sunny_day/log.csv'
IMG_DIR       = 'udacity_dataset_lake_dave/jungle_sunny_day/image'
BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 0.0001
ALPHA_STEER   = 0.8 # steering weight in loss
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42
SAVE_DIR      = 'models'

os.makedirs(SAVE_DIR, exist_ok=True)

# Load & split metadata
train_df, val_df = load_dataframes(CSV_PATH, val_split=VAL_SPLIT, random_seed=RANDOM_SEED)

# Build tf.data pipelines
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

# Instantiate model
model = build_model(input_shape=(66, 200, 3))

# Weighted MSE loss
def weighted_mse(y_true, y_pred):
    steer_true    = y_true[:, 0]
    throttle_true = y_true[:, 1]
    steer_pred    = y_pred[:, 0]
    throttle_pred = y_pred[:, 1]
    loss_s = tf.reduce_mean(tf.square(steer_true - steer_pred))
    loss_t = tf.reduce_mean(tf.square(throttle_true - throttle_pred))
    return ALPHA_STEER * loss_s + (1 - ALPHA_STEER) * loss_t

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=weighted_mse,
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 8) Save final model
model.save(os.path.join(SAVE_DIR, 'final_model.h5'))
print(f"Training complete. Models saved to {SAVE_DIR}")

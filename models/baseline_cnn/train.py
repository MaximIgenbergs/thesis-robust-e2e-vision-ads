import os
import tensorflow as tf
import datetime
from utils.data_loader import DrivingDataset
from utils.utils import load_dataframes
from model import build_model
from matplotlib import pyplot as plt

# Configuration
batch_size = 64
epochs = 100
patience = 7
learning_rate = 0.0001
alpha_steer = 0.8  # steering weight in loss
val_split = 0.2
random_seed = 42

# Paths
LOG_PATH = 'udacity_dataset_lake_dave/jungle_sunny_day/log.csv'
IMG_DIR = 'udacity_dataset_lake_dave/jungle_sunny_day/image'
MODELS_DIR = 'models'
FIG_DIR = 'figures'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Load & split metadata
train_df, val_df = load_dataframes(LOG_PATH, val_split=val_split, random_seed=random_seed)

# Build tf.data pipelines
train_ds = DrivingDataset(
    df=train_df,
    img_dir=IMG_DIR,
    batch_size=batch_size,
    shuffle=True
).dataset()

val_ds = DrivingDataset(
    df=val_df,
    img_dir=IMG_DIR,
    batch_size=batch_size,
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
    return alpha_steer * loss_s + (1 - alpha_steer) * loss_t

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=weighted_mse,
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=patience, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

# Save final model
model.save(os.path.join(MODELS_DIR, 'final_model.h5'))
print(f"Training complete. Models saved to {MODELS_DIR}")

# Create a loss graph
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

ts = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
plot = f"loss_curve_{ts}.png"
PLOT_PATH = os.path.join(FIG_DIR, plot)

plt.savefig(PLOT_PATH)
plt.close()

print(f"Loss curve saved to {PLOT_PATH}")

from __future__ import annotations
import json, datetime, sys
from pathlib import Path

# add project root to path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

from sims.udacity.models.dave2_v2.config import (
    INPUT_SHAPE, ALPHA_STEER, LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE,
    VAL_SPLIT, RANDOM_SEED, AUGMENTATIONS, TRAIN_DIR
)
from sims.udacity.models.dave2_v2.model import build_model
from sims.udacity.models.dave2_v2.utils.data import DrivingDataset

# ---- helpers ----

def load_dataframes(log_path: Path, val_split: float, random_seed: int):
    df = pd.read_csv(log_path)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_size = int(len(df) * val_split)
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df

def write_meta(run_dir: Path, cfg: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "framework": "tensorflow-keras",
        "input_shape": list(cfg["input_shape"]),
        "seed": cfg["seed"],
        "val_split": cfg["val_split"],
        "optimizer": {"type": "adam", "lr": cfg["lr"]},
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "alpha_steer": cfg["alpha_steer"],
        "augmentations": cfg["augmentations"],
        "data": {
            "log_csv": str(TRAIN_LOG_PATH),
            "image_dir": str(TRAIN_IMG_DIR)
        }
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# ---- build datasets ----

train_df, val_df = load_dataframes(
    Path(TRAIN_LOG_PATH),
    val_split=VAL_SPLIT,
    random_seed=RANDOM_SEED
)

train_ds = DrivingDataset(
    df=train_df,
    img_dir=str(TRAIN_IMG_DIR),
    batch_size=BATCH_SIZE,
    shuffle=True
).dataset()

val_ds = DrivingDataset(
    df=val_df,
    img_dir=str(TRAIN_IMG_DIR),
    batch_size=BATCH_SIZE,
    shuffle=False
).dataset()

model = build_model(input_shape=INPUT_SHAPE)

def weighted_mse(y_true, y_pred):
    steer_true, throttle_true = y_true[:, 0], y_true[:, 1]
    steer_pred, throttle_pred = y_pred[:, 0], y_pred[:, 1]
    loss_s = tf.reduce_mean(tf.square(steer_true - steer_pred))
    loss_t = tf.reduce_mean(tf.square(throttle_true - throttle_pred))
    return ALPHA_STEER * loss_s + (1.0 - ALPHA_STEER) * loss_t

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=weighted_mse,
              metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])

RUN_ID = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
RUN_DIR = TRAINING_RUNS_DIR / RUN_ID
BEST_PATH = RUN_DIR / "best_model.h5"

write_meta(RUN_DIR, cfg={
    "input_shape": INPUT_SHAPE,
    "seed": RANDOM_SEED,
    "val_split": VAL_SPLIT,
    "lr": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "alpha_steer": ALPHA_STEER,
    "augmentations": AUGMENTATIONS,
})

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=PATIENCE, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_PATH),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(str(RUN_DIR / "history.csv")),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---- generate loss curve ----

plt.figure()
plt.plot(history.history.get('loss', []),     label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.tight_layout()
loss_curve_path = RUN_DIR / "loss_curve.png"
plt.savefig(loss_curve_path)
plt.close()

print(f"[train] Best model: {BEST_PATH}")
print(f"[train] Loss curve: {loss_curve_path}")
print(f"[train] History CSV: {RUN_DIR / 'history.csv'}")
print(f"[train] Meta: {RUN_DIR / 'meta.json'}")

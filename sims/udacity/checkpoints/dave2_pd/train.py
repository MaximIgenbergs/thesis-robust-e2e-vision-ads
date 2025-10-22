from __future__ import annotations
import math
import tensorflow as tf
from pathlib import Path

from .config import (
    INPUT_SHAPE, NUM_OUTPUTS, INPUTS_GLOB, VAL_SPLIT, RANDOM_SEED,
    LEARNING_RATE, ALPHA_STEER, BATCH_SIZE, EPOCHS, PATIENCE,
    AUGMENTATIONS, RUNS_SUBDIR, RUN_NAME_PREFIX, SAVE_BEST_FILENAME
)
from .model import build_model
from .utils.data import make_filelists, data_generator
from sims.udacity.checkpoints.common.run_io import resolve_runs_dir, make_run_dir, write_meta, loss_plot

def weighted_mse_factory(alpha: float, num_out: int):
    if num_out == 1:
        def mse1(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
        return mse1
    def wmse(y_true, y_pred):
        s_true, t_true = y_true[:, 0], y_true[:, 1]
        s_pred, t_pred = y_pred[:, 0], y_pred[:, 1]
        loss_s = tf.reduce_mean(tf.square(s_true - s_pred))
        loss_t = tf.reduce_mean(tf.square(t_true - t_pred))
        return alpha * loss_s + (1.0 - alpha) * loss_t
    return wmse

def main():
    # --- dataset file lists (deterministic split) ---
    train_files, val_files = make_filelists(INPUTS_GLOB, val_split=VAL_SPLIT)
    if not train_files:
        raise SystemExit(f"No training images found for glob: {INPUTS_GLOB}")

    # --- run directory & artifact paths ---
    runs_base = resolve_runs_dir(model_key=RUN_NAME_PREFIX, fallback_subdir=RUNS_SUBDIR, file=__file__)
    run_dir   = make_run_dir(runs_base, model_key=RUN_NAME_PREFIX)
    best_path = run_dir / SAVE_BEST_FILENAME
    hist_csv  = run_dir / "history.csv"

    # --- model & compile (weighted regression if 2 outputs) ---
    model = build_model(num_outputs=NUM_OUTPUTS)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = weighted_mse_factory(ALPHA_STEER, NUM_OUTPUTS)
    model.compile(optimizer=opt, loss=loss_fn, metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    model.summary()

    steps_per_epoch = max(1, math.floor(len(train_files) / BATCH_SIZE))
    val_steps       = max(1, math.floor(len(val_files)   / BATCH_SIZE))

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_path), monitor="val_loss",
                                           save_best_only=True, save_weights_only=False, verbose=1),
        tf.keras.callbacks.CSVLogger(str(hist_csv)),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        data_generator(train_files, batch_size=BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator(val_files, batch_size=BATCH_SIZE),
        validation_steps=val_steps,
        epochs=EPOCHS,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    # --- artifacts ---
    meta = {
        "created_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "framework": "tensorflow-keras",
        "input_shape": list(INPUT_SHAPE),
        "seed": RANDOM_SEED,
        "val_split": VAL_SPLIT,
        "optimizer": {"type": "adam", "lr": float(LEARNING_RATE)},
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "alpha_steer": float(ALPHA_STEER),
        "augmentations": AUGMENTATIONS,
        "data": {"inputs_glob": INPUTS_GLOB}
    }
    write_meta(run_dir, meta)
    loss_png = loss_plot(history, run_dir)

    print(f"[train:dave2] Best model: {best_path}")
    print(f"[train:dave2] Loss curve: {loss_png}")
    print(f"[train:dave2] History CSV: {hist_csv}")
    print(f"[train:dave2] Meta: {run_dir/'meta.json'}")

if __name__ == "__main__":
    main()

from __future__ import annotations
import math
import tensorflow as tf
import sys
from pathlib import Path

# add project root to path
sys.path.append(str(Path(__file__).resolve().parents[4]))

from sims.udacity.models.dave2.model import build_model
from sims.udacity.models.dave2.utils.data import make_filelists, data_generator
from sims.udacity.logging.training_runs import make_run_dir, write_meta, loss_plot
from sims.udacity.models.dave2.config import (
    MAP_NAME, MODEL_NAME, INPUTS_GLOB, INPUT_SHAPE, NUM_OUTPUTS, LEARNING_RATE, ALPHA_STEER, VAL_SPLIT, RANDOM_SEED,
    BATCH_SIZE, EPOCHS, PATIENCE, AUGMENTATIONS
)

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
    train_files, val_files = make_filelists(INPUTS_GLOB, val_split=VAL_SPLIT)
    if not train_files:
        raise SystemExit(f"[{MODEL_NAME}] No training images found for glob: {INPUTS_GLOB}")

    # <CKPTS_DIR>/<MODEL_NAME>/<MAP_NAME>_<timestamp>/
    run_dir  = make_run_dir(model_key=MODEL_NAME, map_name=MAP_NAME)
    best_path = run_dir / "best_model.h5"
    hist_csv  = run_dir / "history.csv"

    model = build_model(num_outputs=NUM_OUTPUTS)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = weighted_mse_factory(ALPHA_STEER, NUM_OUTPUTS)
    model.compile(optimizer=opt, loss=loss_fn, metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    model.summary()

    n_train = len(train_files)
    n_val   = len(val_files)
    # use ceil so we cover all samples even if not divisible by batch size
    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    val_steps = max(1, math.ceil(n_val / BATCH_SIZE))

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_path), monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1),
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

    meta = {
        "created_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "framework": "tensorflow-keras",
        "model": MODEL_NAME,
        "map": MAP_NAME,
        "checkpoint_path": str(best_path),
        "input_shape": list(INPUT_SHAPE),
        "seed": RANDOM_SEED,
        "val_split": VAL_SPLIT,
        "optimizer": {"type": "adam", "lr": float(LEARNING_RATE)},
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "alpha_steer": float(ALPHA_STEER),
        "augmentations": AUGMENTATIONS,
        "data": {"inputs_glob": INPUTS_GLOB},
        "counts": {"train_images": int(n_train), "val_images": int(n_val)},
    }
    write_meta(run_dir, meta)
    
    loss_png = loss_plot(history, run_dir, title=f"{MODEL_NAME} on {MAP_NAME}")

    print(f"[train:{MODEL_NAME}] Best model: {best_path}")
    print(f"[train:{MODEL_NAME}] Loss curve: {loss_png}")
    print(f"[train:{MODEL_NAME}] History CSV: {hist_csv}")
    print(f"[train:{MODEL_NAME}] Meta: {run_dir/'meta.json'}")

if __name__ == "__main__":
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    main()

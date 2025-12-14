from __future__ import annotations
import math
from datetime import datetime, timezone
import tensorflow as tf  # type: ignore

from scripts.udacity.models.dave2.model import build_dave2
from scripts.udacity.models.dave2.utils.data import make_filelists, data_generator
from scripts.udacity.logging.training_runs import make_run_dir, write_meta, loss_plot
from scripts.udacity.models.dave2.config import MAP_NAME, MODEL_NAME, DATA_DIR, INPUT_SHAPE, NUM_OUTPUTS, LEARNING_RATE, ALPHA_STEER, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE, EPOCHS, PATIENCE, AUGMENTATIONS


def weighted_mse(y_true, y_pred):
    """
    Weighted MSE for steering/throttle regression.
    """
    s_true, t_true = y_true[:, 0], y_true[:, 1]
    s_pred, t_pred = y_pred[:, 0], y_pred[:, 1]

    loss_s = tf.reduce_mean(tf.square(s_true - s_pred))
    loss_t = tf.reduce_mean(tf.square(t_true - t_pred))

    return ALPHA_STEER * loss_s + (1.0 - ALPHA_STEER) * loss_t


def main():
    train_files, val_files = make_filelists(DATA_DIR, val_split=VAL_SPLIT)
    if not train_files:
        raise SystemExit(f"[scripts:train:{MODEL_NAME}][WARN] No training images found in dir: {DATA_DIR}")

    run_dir = make_run_dir(model_key=MODEL_NAME, map_name=MAP_NAME)
    best_path = run_dir / "best_model.h5"
    hist_csv = run_dir / "history.csv"

    model = build_dave2(num_outputs=NUM_OUTPUTS)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss=weighted_mse, metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    model.summary()

    n_train = len(train_files)
    n_val = len(val_files)

    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    val_steps = max(1, math.ceil(n_val / BATCH_SIZE)) if n_val > 0 else 0

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_path), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(hist_csv)),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        data_generator(train_files, batch_size=BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator(val_files, batch_size=BATCH_SIZE) if n_val > 0 else None,
        validation_steps=val_steps if n_val > 0 else None,
        epochs=EPOCHS,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
        "data": {"data_dir": DATA_DIR},
        "counts": {"train_images": int(n_train), "val_images": int(n_val)},
    }

    write_meta(run_dir, meta)

    loss_png = loss_plot(history, run_dir, title=f"{MODEL_NAME} on {MAP_NAME}")

    print(f"[scripts:train:{MODEL_NAME}][INFO] Best model: {best_path}")
    print(f"[scripts:train:{MODEL_NAME}][INFO] Loss curve: {loss_png}")
    print(f"[scripts:train:{MODEL_NAME}][INFO] History CSV: {hist_csv}")
    print(f"[scripts:train:{MODEL_NAME}][INFO] Meta: {run_dir / 'meta.json'}")


if __name__ == "__main__":
    # Allow dynamic GPU memory growth
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    main()

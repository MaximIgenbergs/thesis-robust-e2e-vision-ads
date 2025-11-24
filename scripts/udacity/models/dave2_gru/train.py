from __future__ import annotations
import math
from datetime import datetime, timezone
import tensorflow as tf  # type: ignore

from scripts.udacity.models.dave2_gru.model import build_dave2_gru
from scripts.udacity.models.dave2_gru.utils.data_stream import index_by_track, split_tracks, temporal_split_single_track, total_sequences, make_sequence_dataset
from scripts.udacity.logging.training_runs import make_run_dir, write_meta, loss_plot
from scripts.udacity.models.dave2_gru.config import MAP_NAME, MODEL_NAME, DATA_DIR, INPUT_SHAPE, NUM_OUTPUTS, LEARNING_RATE, ALPHA_STEER, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE, EPOCHS, PATIENCE, AUGMENTATIONS, SEQ_LEN, STRIDE


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
    per_track = index_by_track(DATA_DIR)
    if not per_track:
        raise SystemExit(f"[scripts:train:{MODEL_NAME}][WARN] No images/jsons found in dir: {DATA_DIR}")

    if len(per_track) == 1:
        per_track, train_ids, val_ids = temporal_split_single_track(per_track, seq_len=SEQ_LEN, val_split=VAL_SPLIT)
    else:
        train_ids, val_ids = split_tracks(per_track, VAL_SPLIT, RANDOM_SEED)

    if not train_ids:
        raise SystemExit(
            f"[scripts:train:{MODEL_NAME}][WARN] Train split empty. "
            f"Consider lowering VAL_SPLIT or collecting more data. (Data Directory: {DATA_DIR})"
        )

    ds_train = make_sequence_dataset(per_track, train_ids, SEQ_LEN, STRIDE, INPUT_SHAPE, NUM_OUTPUTS, BATCH_SIZE, repeat=True)
    ds_val = make_sequence_dataset(per_track, val_ids, SEQ_LEN, STRIDE, INPUT_SHAPE, NUM_OUTPUTS, BATCH_SIZE, repeat=True) if val_ids else None

    run_dir = make_run_dir(model_key=MODEL_NAME, map_name=MAP_NAME)
    best_path = run_dir / "best_model.h5"
    hist_csv = run_dir / "history.csv"

    model = build_dave2_gru(INPUT_SHAPE, SEQ_LEN, NUM_OUTPUTS, LEARNING_RATE)

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss=weighted_mse, metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    model.summary()

    n_train = total_sequences(per_track, train_ids, SEQ_LEN, STRIDE)
    n_val = total_sequences(per_track, val_ids, SEQ_LEN, STRIDE) if ds_val else 0

    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    val_steps = max(1, math.ceil(n_val / BATCH_SIZE)) if ds_val else None

    monitor = "val_loss" if ds_val else "loss"

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=PATIENCE, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_path), monitor=monitor, save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(hist_csv)),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
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
        "input_shape": [SEQ_LEN] + list(INPUT_SHAPE),
        "seed": RANDOM_SEED,
        "val_split": VAL_SPLIT,
        "optimizer": {"type": "adam", "lr": float(LEARNING_RATE)},
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "alpha_steer": float(ALPHA_STEER),
        "augmentations": AUGMENTATIONS,
        "data": {"data_dir": DATA_DIR, "seq_len": SEQ_LEN, "stride": STRIDE},
        "counts": {"train_sequences": int(n_train), "val_sequences": int(n_val)},
    }

    write_meta(run_dir, meta)

    loss_png = loss_plot(history, run_dir, title=f"DAVE2-GRU on {MAP_NAME}")

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

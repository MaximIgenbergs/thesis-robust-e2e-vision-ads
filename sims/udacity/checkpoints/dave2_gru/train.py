# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import tensorflow as tf
from pathlib import Path
from sims.udacity.checkpoints.common.run_io import resolve_runs_dir, make_run_dir, write_meta, loss_plot

# Read config (no CLI)
from sims.udacity.checkpoints.dave2_gru.config import (
    INPUT_SHAPE, NUM_OUTPUTS, INPUTS_GLOB, VAL_SPLIT, RANDOM_SEED,
    SEQ_LEN, STRIDE, LEARNING_RATE, ALPHA_STEER,
    BATCH_SIZE, EPOCHS, PATIENCE, AUGMENTATIONS,
    RUNS_SUBDIR, RUN_NAME_PREFIX, SAVE_BEST_FILENAME
)
from sims.udacity.checkpoints.dave2_gru.model import build_dave2_gru
from sims.udacity.checkpoints.dave2_gru.utils.data_stream import (
    index_by_track, split_tracks, total_sequences, make_sequence_dataset
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
    # Index on disk only (paths + small metadata)
    per_track = index_by_track(INPUTS_GLOB)
    if not per_track:
        raise SystemExit(f"[dave2_gru] no images/jsons found for glob: {INPUTS_GLOB}")

    train_ids, val_ids = split_tracks(per_track, VAL_SPLIT, RANDOM_SEED)
    if not train_ids:
        raise SystemExit("[dave2_gru] train split empty. Lower VAL_SPLIT or check data.")

    # Datasets (streaming)
    ds_train = make_sequence_dataset(
        per_track, train_ids, SEQ_LEN, STRIDE, INPUT_SHAPE, NUM_OUTPUTS, BATCH_SIZE,
        repeat=True
    )
    ds_val = make_sequence_dataset(
        per_track, val_ids,   SEQ_LEN, STRIDE, INPUT_SHAPE, NUM_OUTPUTS, BATCH_SIZE,
        repeat=True
    )


    # Steps per epoch from counts (no data loaded)
    n_train = total_sequences(per_track, train_ids, SEQ_LEN, STRIDE)
    n_val   = total_sequences(per_track, val_ids,   SEQ_LEN, STRIDE)
    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    val_steps       = max(1, math.ceil(n_val   / BATCH_SIZE))

    # Run directory & artifacts
    # Write runs into sims/udacity/checkpoints/dave2_gru/training_runs/
    runs_base = Path(__file__).resolve().parent / "training_runs"
    runs_base.mkdir(parents=True, exist_ok=True)

    run_dir   = make_run_dir(runs_base, model_key=RUN_NAME_PREFIX)
    best_path = run_dir / SAVE_BEST_FILENAME
    hist_csv  = run_dir / "history.csv"

    # Model & compile
    model = build_dave2_gru(seq_len=SEQ_LEN, num_outputs=NUM_OUTPUTS, learning_rate=LEARNING_RATE)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = weighted_mse_factory(ALPHA_STEER, NUM_OUTPUTS)
    model.compile(optimizer=opt, loss=loss_fn, metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_path), monitor="val_loss",
                                           save_best_only=True, save_weights_only=False, verbose=1),
        tf.keras.callbacks.CSVLogger(str(hist_csv)),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
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
        "created_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "framework": "tensorflow-keras",
        "input_shape": [SEQ_LEN] + list(INPUT_SHAPE),
        "seed": RANDOM_SEED,
        "val_split": VAL_SPLIT,
        "optimizer": {"type": "adam", "lr": float(LEARNING_RATE)},
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "alpha_steer": float(ALPHA_STEER),
        "augmentations": AUGMENTATIONS,
        "data": {"inputs_glob": INPUTS_GLOB, "seq_len": SEQ_LEN, "stride": STRIDE},
        "counts": {"train_sequences": int(n_train), "val_sequences": int(n_val)}
    }
    write_meta(run_dir, meta)
    loss_png = loss_plot(history, run_dir)

    print(f"[train:dave2_gru] Best model: {best_path}")
    print(f"[train:dave2_gru] Loss curve: {loss_png}")
    print(f"[train:dave2_gru] History CSV: {hist_csv}")
    print(f"[train:dave2_gru] Meta: {run_dir/'meta.json'}")

if __name__ == "__main__":
    # Optional: allow dynamic GPU memory growth (not related to the RAM OOM)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
    main()

from __future__ import annotations
import math
import csv
from datetime import datetime, timezone
import numpy as np
import torch
from tqdm.auto import tqdm

from scripts.udacity.models.vit.model import ViT
from scripts.udacity.models.vit.utils.data import make_filelists, data_generator
from scripts.udacity.logging.training_runs import make_run_dir, write_meta, loss_plot
from scripts.udacity.models.vit.config import MAP_NAME, MODEL_NAME, DATA_DIR, INPUT_SHAPE, NUM_OUTPUTS, LEARNING_RATE, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE, EPOCHS, PATIENCE, AUGMENTATIONS


def main():
    train_files, val_files = make_filelists(DATA_DIR, val_split=VAL_SPLIT)
    if not train_files:
        raise SystemExit(f"[scripts:train:{MODEL_NAME}][WARN] No training images found in dir: {DATA_DIR}")

    run_dir = make_run_dir(model_key=MODEL_NAME, map_name=MAP_NAME)
    best_path = run_dir / "best_model.ckpt"
    hist_csv = run_dir / "history.csv"

    print(f"[scripts:train:{MODEL_NAME}][INFO] Train images: {len(train_files)}, Val images: {len(val_files)}")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[scripts:train:{MODEL_NAME}][INFO] Using device: {device}")

    model = ViT(
        input_shape=(3, INPUT_SHAPE[0], INPUT_SHAPE[1]),
        learning_rate=float(LEARNING_RATE),
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE))

    n_train = len(train_files)
    n_val = len(val_files)

    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    val_steps = max(1, math.ceil(n_val / BATCH_SIZE)) if n_val > 0 else 0

    print(
        f"[scripts:train:{MODEL_NAME}][INFO] steps_per_epoch={steps_per_epoch}, val_steps={val_steps}, batch_size={BATCH_SIZE}"
    )

    gen_train = data_generator(train_files, batch_size=BATCH_SIZE, train=True)
    gen_val = data_generator(val_files, batch_size=BATCH_SIZE, train=False) if n_val > 0 else None

    history = {
        "loss": [],
        "val_loss": [],
        "rmse": [],
        "val_rmse": [],
    }

    best_val_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_count = 0

        model.train()
        for _ in tqdm(range(steps_per_epoch), desc=f"[scripts:train:{MODEL_NAME}][INFO] Epoch {epoch}/{EPOCHS} Training", leave=False):
            x_batch, y_batch = next(gen_train)
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            bs = x_batch.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        if total_count == 0:
            train_loss = math.inf
            train_rmse = math.inf
        else:
            train_loss = total_loss / total_count
            train_rmse = math.sqrt(train_loss)

        if gen_val is not None and val_steps > 0:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for _ in tqdm(range(val_steps), desc=f"[scripts:train:{MODEL_NAME}][INFO] Epoch {epoch}/{EPOCHS} Validation", leave=False):
                    x_val, y_val = next(gen_val)
                    x_val = x_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, y_val)
                    bs_val = x_val.size(0)
                    val_total += loss_val.item() * bs_val
                    val_count += bs_val

            if val_count == 0:
                val_loss = math.inf
                val_rmse = math.inf
            else:
                val_loss = val_total / val_count
                val_rmse = math.sqrt(val_loss)
        else:
            val_loss = train_loss
            val_rmse = train_rmse

        history["loss"].append(train_loss)
        history["rmse"].append(train_rmse)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        print(f"[scripts:train:{MODEL_NAME}][INFO] Epoch {epoch}/{EPOCHS} loss={train_loss:.6f} rmse={train_rmse:.6f} val_loss={val_loss:.6f} val_rmse={val_rmse:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"[scripts:train:{MODEL_NAME}][INFO] Epoch {epoch}/{EPOCHS} New best model saved to {best_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"[scripts:train:{MODEL_NAME}][INFO] Epoch {epoch}/{EPOCHS} Early stopping after {epoch} epochs (no improvement for {PATIENCE} epochs).")
            break

    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "rmse", "val_loss", "val_rmse"])
        for i in range(len(history["loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["loss"][i],
                    history["rmse"][i],
                    history["val_loss"][i],
                    history["val_rmse"][i],
                ]
            )

    history_obj = type("History", (), {})()
    history_obj.history = history
    loss_png = loss_plot(history_obj, run_dir, title=f"{MODEL_NAME} on {MAP_NAME}")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "pytorch",
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
        "loss": "mse",
        "augmentations": AUGMENTATIONS,
        "num_outputs": NUM_OUTPUTS,
        "data": {"data_dir": DATA_DIR},
        "counts": {"train_images": int(n_train), "val_images": int(n_val)},
    }

    write_meta(run_dir, meta)

    print(f"[scripts:train:{MODEL_NAME}][INFO] Best model: {best_path}")
    print(f"[scripts:train:{MODEL_NAME}][INFO] Loss curve: {loss_png}")
    print(f"[scripts:train:{MODEL_NAME}][INFO] History CSV: {hist_csv}")
    print(f"[scripts:train:{MODEL_NAME}][INFO] Meta: {run_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

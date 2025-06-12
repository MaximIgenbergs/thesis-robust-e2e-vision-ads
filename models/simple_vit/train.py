import sys
import pathlib
import datetime
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Set multiprocessing to fork for macOS compatibility
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Add project root to PYTHONPATH
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.utils.utils import load_dataframes
from models.utils.paths import get_model_dir, get_fig_dir, TRAIN_IMG_DIR, TRAIN_LOG_PATH
from models.utils.training_defaults import BATCH_SIZE, EPOCHS, PATIENCE, VAL_SPLIT, RANDOM_SEED
from models.utils.device_config import ACCELERATOR, DEVICE
from models.simple_vit.model import ViT

class DrivingDataset(Dataset):
    """Simple dataset for [steering, throttle] regression."""
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.frame_col = 'frame' if 'frame' in df.columns else 'image_filename'
        self.steer_col = 'steering' if 'steering' in df.columns else 'predicted_steering_angle'
        self.thr_col   = 'throttle' if 'throttle' in df.columns else 'predicted_throttle'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row[self.frame_col]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        target = torch.tensor([row[self.steer_col], row[self.thr_col]], dtype=torch.float32)
        return img, target

# match style of other training scripts
pl.seed_everything(RANDOM_SEED)
torch.set_float32_matmul_precision('high')

# load & split data
train_df, val_df = load_dataframes(TRAIN_LOG_PATH, val_split=VAL_SPLIT, random_seed=RANDOM_SEED)

transform = T.Compose([T.Resize((160, 320)), T.ToTensor()])

train_loader = DataLoader(
    DrivingDataset(train_df, TRAIN_IMG_DIR, transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)
val_loader = DataLoader(
    DrivingDataset(val_df, TRAIN_IMG_DIR, transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)

# prepare model, callbacks, trainer
model = ViT()
csv_logger = CSVLogger(save_dir=get_fig_dir('vit'), name='logs', version='vit')
checkpoint_cb = pl.callbacks.ModelCheckpoint(
    dirpath=get_model_dir('vit'), filename='best_model', monitor='val/loss', save_top_k=1, mode='min', verbose=True
)
earlystop_cb = pl.callbacks.EarlyStopping(
    monitor='val/loss', mode='min', patience=PATIENCE, verbose=True
)
trainer = pl.Trainer(
    accelerator=ACCELERATOR, devices=[DEVICE], max_epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb], logger=csv_logger
)

# train & save
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint(get_model_dir('vit') / 'final_model.ckpt')
print(f"Training complete. Checkpoints in: {get_model_dir('vit')}")

# plot & save loss curve
metrics_file = Path(csv_logger.log_dir) / 'metrics.csv'
if metrics_file.exists():
    df = pd.read_csv(metrics_file).dropna(subset=['epoch','train/loss','val/loss'])
    plt.figure()
    plt.plot(df['epoch'], df['train/loss'], label='train_loss')
    plt.plot(df['epoch'], df['val/loss'],   label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    out = get_fig_dir('vit') / f"loss_curve_{ts}.png"
    plt.savefig(out)
    plt.close()
    print(f"Loss curve saved to: {out}")

# cleanup: remove Lightning log artifacts
import shutil
logs_root = Path(csv_logger.save_dir) / csv_logger.name
if logs_root.exists():
    shutil.rmtree(logs_root)

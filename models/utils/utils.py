import datetime
import pandas as pd
from typing import Tuple, Union
import pathlib
from paths import PROJECT_DIR

COLLECTION_ROOT = PROJECT_DIR / 'data' / 'collections'
COLLECTION_ROOT.mkdir(parents=True, exist_ok=True)

def make_collection_dir(collector: str) -> pathlib.Path:
    """
    Creates a new folder under data/collections named
    <collector>_<YYYYMMDD>T<HHMMSS>, with subdirectories:
      - image/
      - segmentation/
    and empty files:
      - log.csv
      - info.csv

    Returns the Path to the created folder.
    """
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    folder = COLLECTION_ROOT / f"{collector}_{ts}"
    # create subdirectories
    (folder / 'image').mkdir(parents=True, exist_ok=True)
    (folder / 'segmentation').mkdir(parents=True, exist_ok=True)
    # create empty log and info files
    (folder / 'log.csv').touch()
    (folder / 'info.csv').touch()
    return folder


def load_dataframes(
    log_path: Union[str, pathlib.Path],
    val_split: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads a CSV at log_path, shuffles, and splits into train/val DataFrames.

    Returns:
        train_df, val_df
    """
    df = pd.read_csv(log_path)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_size = int(len(df) * val_split)
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    return train_df, val_df
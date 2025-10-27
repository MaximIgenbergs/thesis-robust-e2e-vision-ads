import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[6]
COLLECTION_ROOT = PROJECT_DIR / 'data' / 'collections'

# Name of the specific collection folder to train on.
TRAIN_COLLECTION_NAME = "pid_20250602T130112"

# Derived paths for training data
TRAIN_DATA_DIR = COLLECTION_ROOT / TRAIN_COLLECTION_NAME
TRAIN_IMG_DIR  = TRAIN_DATA_DIR   / 'image'
TRAIN_LOG_PATH = TRAIN_DATA_DIR   / 'log.csv'

if not TRAIN_DATA_DIR.exists():
    raise FileNotFoundError(
        f"Training data directory not found: {TRAIN_DATA_DIR}\n"
        "Please set TRAIN_COLLECTION_NAME in utils/paths.py to a valid folder under data/collections/"
    )

def get_model_dir(model_name: str) -> pathlib.Path:
    """
    Returns the path where model checkpoints for a given model are stored:
    PROJECT_DIR/models/<model_name>/models
    """
    path = PROJECT_DIR / 'models' / model_name / 'models'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_fig_dir(model_name: str) -> pathlib.Path:
    """
    Returns the path where figures for a given model are saved:
    PROJECT_DIR/models/<model_name>/figures
    """
    path = PROJECT_DIR / 'models' / model_name / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    return path
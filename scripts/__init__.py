from pathlib import Path
import sys
from typing import Union, Optional
import yaml

ROOT = Path("/home/maximigenbergs/thesis-robust-e2e-vision-ads")
CKPTS_DIR = ROOT / "checkpoints"

PD_DIR = ROOT / "external" / "perturbation-drive"
if str(PD_DIR) not in sys.path: 
    sys.path.insert(0, str(PD_DIR))

TCP_ROOT = ROOT / "external" / "TCP" # TODO: could be an issue if loaded every time and wrong venv is selected that doesnt contain anything to do with TCP. lets see. 
if str(TCP_ROOT) not in sys.path:
    sys.path.insert(0, str(TCP_ROOT))

def abs_path(p: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path against the repository root.

    - No argument or None returns ROOT.
    - Absolute paths are returned as-is.
    - Relative paths are resolved as ROOT / p.
    """
    if p is None:
        return ROOT

    if isinstance(p, Path):
        path = p
    elif isinstance(p, str):
        path = Path(p)
    else:
        raise TypeError(f"abs_path() only accepts str | Path | None, got {type(p).__name__}")

    return path if path.is_absolute() else (ROOT / path).resolve()

def load_cfg(cfg_path: Optional[Union[str, Path]] = None) -> dict:
    if isinstance(cfg_path, str):
        cfg_path = Path(cfg_path)

    with abs_path(cfg_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
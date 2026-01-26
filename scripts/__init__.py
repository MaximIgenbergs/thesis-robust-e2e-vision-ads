from pathlib import Path
import sys
from typing import Union, Optional
import yaml

ROOT = Path(__file__).parent.parent.resolve()
CKPTS_DIR = ROOT / "checkpoints"

# add perturbationdrive to path 
PD_DIR = ROOT / "external" / "perturbation-drive"
if str(PD_DIR) not in sys.path: 
    sys.path.insert(0, str(PD_DIR))

# add tcp to path
TCP_ROOT = ROOT / "external" / "TCP" 
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
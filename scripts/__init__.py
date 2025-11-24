from pathlib import Path
import sys
from typing import Union, Optional

ROOT = Path(__file__).resolve().parents[1]
CKPTS_DIR = ROOT / "scripts" / "udacity" / "checkpoints"

PD_DIR = ROOT / "external" / "perturbation-drive"
if str(PD_DIR) not in sys.path: 
    sys.path.insert(0, str(PD_DIR))

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
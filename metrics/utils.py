from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def safe_str(x: Any, default: str = "unknown") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def first_present(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

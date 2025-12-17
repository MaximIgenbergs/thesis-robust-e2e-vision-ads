from pathlib import Path
from typing import Any, Dict, List
import yaml


def load_roads(yaml_path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"roads.yaml not found at: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    roads_def = cfg.get("roads") or {}
    sets_def = cfg.get("sets") or {}

    if not roads_def:
        raise ValueError(f"No 'roads' section in {yaml_path}")
    if not sets_def:
        raise ValueError(f"No 'sets' section in {yaml_path}")

    # Basic validation
    for name, spec in roads_def.items():
        angles = spec.get("angles")
        segs = spec.get("segs")
        if not isinstance(angles, list) or not isinstance(segs, list):
            raise ValueError(f"[road:{name}] 'angles' and 'segs' must be lists (file: {yaml_path})")
        if len(angles) != len(segs):
            raise ValueError(f"[road:{name}] length mismatch: angles({len(angles)}) != segs({len(segs)}) (file: {yaml_path})")

    return roads_def, sets_def
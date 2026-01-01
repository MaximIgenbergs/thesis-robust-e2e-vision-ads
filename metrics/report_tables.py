from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from metrics.utils import ensure_dir


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_csv_ordered(path: Path, rows: List[Dict[str, Any]], keys: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_csv_multiheader(path: Path, header_rows: Sequence[Sequence[str]], data_rows: Sequence[Sequence[Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for hr in header_rows:
            w.writerow(list(hr))
        for dr in data_rows:
            w.writerow(list(dr))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

from __future__ import annotations

from typing import Any, Dict, List, Optional

FAIL_EVENTS = {"collision", "out_of_track"}


def has_failure_for_task(*, events: List[Dict[str, Any]] | None, task_id: str) -> bool:
    if not events:
        return False
    tid = str(task_id)
    for e in events:
        if not isinstance(e, dict):
            continue
        if str(e.get("segment_id", "")) != tid:
            continue
        name = str(e.get("event", "")).strip()
        if name in FAIL_EVENTS:
            return True
    return False


def first_failure_step_for_task(*, events: List[Dict[str, Any]] | None, task_id: str) -> Optional[int]:
    if not events:
        return None

    best: Optional[int] = None
    tid = str(task_id)
    for e in events:
        if not isinstance(e, dict):
            continue
        if str(e.get("segment_id", "")) != tid:
            continue
        name = str(e.get("event", "")).strip()
        if name not in FAIL_EVENTS:
            continue
        try:
            step = int(e.get("step"))
        except Exception:
            continue
        if best is None or step < best:
            best = step
    return best


def first_failure_step_from_pd_steps(steps: List[Dict[str, Any]] | None) -> Optional[int]:
    if not steps:
        return None

    for s in steps:
        if not isinstance(s, dict):
            continue
        try:
            k = int(s.get("step", -1))
        except Exception:
            k = -1

        info = s.get("info") if isinstance(s.get("info"), dict) else {}

        is_success = info.get("is_success", None)
        if isinstance(is_success, (int, float)) and int(is_success) < 0:
            return k if k >= 0 else None

        track = info.get("track", None)
        if track is not None:
            t = str(track).lower()
            if any(x in t for x in ["out", "off", "oob", "collision", "crash", "block"]):
                return k if k >= 0 else None

        for flag in ["collision", "collided", "out_of_track", "off_road", "blocked"]:
            v = info.get(flag, None)
            if isinstance(v, bool) and v:
                return k if k >= 0 else None

    return None


def infer_stop_idx(*, series_len: int, event_step: Optional[int]) -> int:
    """Returns the index of the last successful step + 1."""
    if series_len <= 0:
        return 0
    if event_step is None:
        return series_len
    try:
        s = int(event_step)
    except Exception:
        return series_len
    return max(0, min(series_len, s))

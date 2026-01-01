from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from external.udacity_gym import UdacityAction
from external.udacity_gym.agent import PIDUdacityAgent_Angle


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip()
        return float(x)
    except Exception:
        return default


def _pick_seq(d: Dict[str, Any], keys: Sequence[str]) -> List[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v:
            out: List[float] = []
            for item in v:
                out.append(_to_float(item, default=0.0))
            return out
    return []


def _pick_speed_seq(entry: Dict[str, Any], n: int) -> List[float]:
    seq = _pick_seq(entry, ["speeds", "speed", "vel", "velocity"])
    if not seq:
        # if speed is logged as a scalar somewhere, expand it
        for k in ["speed", "vel", "velocity"]:
            if k in entry and not isinstance(entry[k], list):
                return [_to_float(entry[k], 0.0)] * n
        return [0.0] * n
    if len(seq) < n:
        # pad by last value
        last = seq[-1]
        seq = seq + [last] * (n - len(seq))
    return seq[:n]


@dataclass
class ObsProxy:
    """
    Minimal object passed to PIDUdacityAgent_Angle.
    We provide common fields used by PID agents: cte, angle, speed, time.
    Any missing attribute access returns 0.0 to avoid crashes.
    """
    cte: float
    angle: float
    speed: float
    time: float = 0.0
    sector: int = 0

    def __getattr__(self, name: str) -> Any:
        # conservative default for unknown numeric attributes
        return 0.0


def _pid_action_to_pair(a: Any) -> Optional[List[float]]:
    if isinstance(a, UdacityAction):
        return [float(a.steering_angle), float(a.throttle)]
    if isinstance(a, (list, tuple)) and len(a) >= 2:
        return [float(a[0]), float(a[1])]
    return None


def backfill_log_file(
    log_path: Path,
    *,
    target_speed: float,
    track: str,
    dt: float,
    inplace: bool,
) -> Tuple[int, int]:
    data = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return 0, 0

    agent = PIDUdacityAgent_Angle(
        target_speed=float(target_speed),
        track=str(track),
        before_action_callbacks=[],
        after_action_callbacks=[],
    )

    changed = 0
    skipped = 0

    for entry in data:
        if not isinstance(entry, dict):
            continue

        pid_actions = entry.get("pid_actions")
        if isinstance(pid_actions, list) and len(pid_actions) > 0:
            skipped += 1
            continue

        xte = _pick_seq(entry, ["xte", "cte"])
        ang = _pick_seq(entry, ["angle_diff", "angle_errors", "angle_err", "angle"])
        n = min(len(xte), len(ang)) if (xte and ang) else max(len(xte), len(ang))

        if n <= 0:
            # nothing to compute
            entry["pid_actions"] = []
            changed += 1
            continue

        xte = xte[:n] if xte else [0.0] * n
        ang = ang[:n] if ang else [0.0] * n
        spd = _pick_speed_seq(entry, n=n)

        out_pid: List[List[float]] = []
        for i in range(n):
            obs = ObsProxy(cte=xte[i], angle=ang[i], speed=spd[i], time=float(i) * float(dt))
            a = agent(obs)
            pair = _pid_action_to_pair(a)
            if pair is None:
                pair = [0.0, 0.0]
            out_pid.append(pair)

        entry["pid_actions"] = out_pid
        changed += 1

    if inplace:
        log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        out_path = log_path.with_suffix(".with_pid.json")
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return changed, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--target-speed", type=float, default=22.0)
    ap.add_argument("--track", type=str, default="jungle")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    log_files = sorted(run_dir.glob("episodes/*/log.json"))
    if not log_files:
        raise SystemExit(f"no episodes/*/log.json under: {run_dir}")

    total_changed = 0
    total_skipped = 0
    for lp in log_files:
        ch, sk = backfill_log_file(
            lp,
            target_speed=args.target_speed,
            track=args.track,
            dt=args.dt,
            inplace=bool(args.inplace),
        )
        total_changed += ch
        total_skipped += sk

    mode = "inplace" if args.inplace else "copy"
    print(f"[OK] backfilled pid_actions ({mode}) in {len(log_files)} files: changed={total_changed}, skipped_existing={total_skipped}")


if __name__ == "__main__":
    main()

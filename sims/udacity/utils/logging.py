"""
Udacity sim run logging utilities.

Output Structure
-------
<runs_root>/<YYYYmmdd_HHMMSS>__<ckpt_name>/
  ├── manifest.json           # run-level metadata + episode index
  ├── config_snapshot.json    # paths/roads/perturbations/run snapshot
  ├── env_snapshot.txt        # pip freeze
  └── episodes/
      └── 0001/
          ├── meta.json       # episode metadata
          └── pd_log.json     # PerturbationDrive log
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


# ---- small helpers ----

def _dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_tag(tag: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in tag)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-_") or "run"


def make_run_id(tag: str) -> str:
    """YYYYmmdd_HHMMSS__<tag> (tag sanitized)."""
    return f"{_ts_now()}__{_sanitize_tag(tag)}"


def make_run_dir(runs_root: Union[str, Path], run_id: str) -> Path:
    """Create run directory structure: <runs_root>/<run_id>/{episodes/}."""
    root = Path(runs_root).expanduser().resolve()
    run_dir = root / run_id
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


def module_public_dict(mod) -> Dict[str, Any]:
    """Export non-callable, non-underscore attributes from a module as JSON-serializable dict."""
    out: Dict[str, Any] = {}
    for k, v in vars(mod).items():
        if k.startswith("_") or callable(v):
            continue
        try:
            json.dumps(v)  # type: ignore[arg-type]
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out


def best_effort_git_sha(repo_root: Union[str, Path]) -> Optional[str]:
    """Return current HEAD SHA or None if not a git repo."""
    try:
        r = subprocess.run(
            ["git", "-C", str(Path(repo_root)), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return r.stdout.strip()
    except Exception:
        return None


def pip_freeze() -> str:
    """Return `pip freeze` output as a string; empty string on failure."""
    try:
        r = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        return r.stdout
    except Exception:
        return ""


# ---- run logger ----

@dataclass
class PlatformInfo:
    system: str
    release: str
    python: str


class RunLogger:
    """
    Manage a run directory with:
      - manifest.json (run-level metadata + episode index)
      - config_snapshot.json (paths/roads/perturbations/run)
      - env_snapshot.txt (pip freeze)
      - episodes/<id>/{meta.json, ...}

    Usage:
        run_id   = make_run_id(ckpt_name)  # or model tag
        run_dir  = make_run_dir(runs_root, run_id)
        logger   = RunLogger(run_dir, model="dave2", checkpoint="...", sim_name="udacity", git_info={...})
        logger.snapshot_configs(sim_app, ckpt, cfg_paths, cfg_roads, cfg_pert, cfg_run)
        logger.snapshot_env()
        eid, ep_dir = logger.new_episode(1, meta)
        # ... produce pd_log.json into ep_dir ...
        logger.complete_episode(eid, status="ok", wall_time_s=12.345)
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        model: str,
        checkpoint: Optional[Union[str, Path]],
        sim_name: str,
        git_info: Dict[str, Any],
    ):
        self.run_dir = Path(run_dir).resolve()
        self.ep_root = self.run_dir / "episodes"
        self.ep_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"

        plat = PlatformInfo(platform.system(), platform.release(), platform.python_version())
        self.manifest: Dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_dir.name,
            "timestamp": datetime.now().isoformat(),
            "host": platform.node(),
            "platform": {"system": plat.system, "release": plat.release, "python": plat.python},
            "model": model,
            "checkpoint": (str(checkpoint) if checkpoint is not None else None),
            "sim": sim_name,
            "git": git_info,
            "episodes": [],  # list of dicts
        }
        _dump_json(self.manifest_path, self.manifest)

    # --- snapshots ---

    def snapshot_configs(
        self,
        sim_app: Union[str, Path],
        ckpt: Optional[Union[str, Path]],
        cfg_paths: Dict[str, Any],
        cfg_roads: Dict[str, Any],
        cfg_perturbations: Dict[str, Any],
        cfg_run: Dict[str, Any],
    ) -> None:
        snap = {
            "paths": {
                "sim_app": str(sim_app),
                "checkpoint": (str(ckpt) if ckpt else None),
            },
            "configs": {
                "paths": cfg_paths,
                "roads": cfg_roads,
                "perturbations": cfg_perturbations,
                "run": cfg_run,
            },
        }
        _dump_json(self.run_dir / "config_snapshot.json", snap)

    def snapshot_env(self, content: Optional[str] = None) -> None:
        text = content if isinstance(content, str) else pip_freeze()
        (self.run_dir / "env_snapshot.txt").write_text(text)

    # --- episodes ---

    def new_episode(self, idx: int, meta: Dict[str, Any]) -> Tuple[str, Path]:
        eid = f"{idx:04d}"
        ep_dir = self.ep_root / eid
        ep_dir.mkdir(parents=True, exist_ok=True)
        _dump_json(ep_dir / "meta.json", meta)
        self.manifest["episodes"].append({
            "id": eid,
            "road": meta.get("road"),
            "perturbation": meta.get("perturbation"),
            "severity": meta.get("severity"),
            "log": f"episodes/{eid}/pd_log.json",
            "status": "pending",
        })
        _dump_json(self.manifest_path, self.manifest)
        return eid, ep_dir

    def complete_episode(self, eid: str, status: str, wall_time_s: float) -> None:
        for e in self.manifest.get("episodes", []):
            if e.get("id") == eid:
                e["status"] = status
                e["wall_time_s"] = round(float(wall_time_s), 3)
                break
        _dump_json(self.manifest_path, self.manifest)

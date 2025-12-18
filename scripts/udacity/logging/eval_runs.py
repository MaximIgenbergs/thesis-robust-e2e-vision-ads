"""
Creates logs for evaluation runs on Udacity maps.

Output structure:
  <ROOT>/runs/<map>/<test_type>/<model>_<YYYYmmdd_HHMMSS>/
    ├── manifest.json           # run-level metadata + episode index
    ├── config_snapshot.json    # paths, roads, perturbations, run etc...
    ├── env_snapshot.txt        # pip freeze
    └── episodes/
        └── 0001/
            ├── meta.json       # episode metadata
            └── log.json        # complete log of the episode
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def dump_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def make_run_id(model_name: str) -> str:
    """
    Return '<model_name>_<YYYYmmdd_HHMMSS>'.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{ts}"


def make_run_dir(runs_root: Path, run_id: str) -> Path:
    """
    Create <runs_root>/<run_id>/episodes and return the run_dir path.
    """
    run_dir = runs_root / run_id
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_run_dir(model_name: str, runs_root: Path | str) -> tuple[str, Path]:
    """
    Create a fresh run directory under <ROOT>/runs.

    Folder name: '<model_name>_<YYYYmmdd_HHMMSS>'.

    Returns:
        (run_id, run_dir)
    """
    runs_root_path = Path(runs_root)
    runs_root_path.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id(model_name)
    run_dir = make_run_dir(runs_root_path, run_id)
    return run_id, run_dir



def best_effort_git_sha(repo_root: Path | str) -> str | None:
    """
    Return the current HEAD SHA.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repo_root)), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def pip_freeze() -> str:
    """
    Return 'pip freeze' output as a string.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception:
        return ""


@dataclass
class PlatformInfo:
    system: str
    release: str
    python: str


class RunLogger:
    """
    Manage a run directory with:
      - manifest.json
      - config_snapshot.json
      - env_snapshot.txt
      - episodes/<id>/{meta.json, log.json}
    """

    def __init__(self, run_dir: Path | str, model: str, checkpoint: Path | str | None, sim_name: str, git_info: dict[str, Any]) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.ep_root = self.run_dir / "episodes"
        self.ep_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"

        plat = PlatformInfo(platform.system(), platform.release(), platform.python_version())
        self.manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_dir.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "platform": {
                "system": plat.system,
                "release": plat.release,
                "python": plat.python,
            },
            "model": model,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "sim": sim_name,
            "git": git_info,
            "episodes": [],
        }
        dump_json(self.manifest_path, self.manifest)

    def snapshot_configs(
        self,
        sim_app: Path | str,
        ckpt: Path | str | None,
        cfg_logging: dict[str, Any] | None = None,
        cfg_udacity: dict[str, Any] | None = None,
        cfg_models: dict[str, Any] | None = None,
        cfg_roads: dict[str, Any] | None = None,
        cfg_perturbations: dict[str, Any] | None = None,
        cfg_segments: dict[str, Any] | None = None,
        cfg_scenarios: dict[str, Any] | None = None,
        cfg_run: dict[str, Any] | None = None,
        cfg_host_port: dict[str, Any] | None = None,
    ) -> None:
        """
        Write config_snapshot.json with sim path + config blocks.
        """
        snap: dict[str, Any] = {
            "paths": {
                "sim_app": str(sim_app),
                "checkpoint": str(ckpt) if ckpt else None,
            },
            "configs": {
                "logging": cfg_logging,
                "udacity": cfg_udacity,
                "models": cfg_models,
                "roads": cfg_roads,
                "perturbations": cfg_perturbations,
                "segments": cfg_segments,
                "scenarios": cfg_scenarios,
                "run": cfg_run,
                "host_port": cfg_host_port,
            },
        }
        dump_json(self.run_dir / "config_snapshot.json", snap)

    def snapshot_env(self, content: str) -> None:
        """
        Write env_snapshot.txt (expects precomputed content).
        """
        (self.run_dir / "env_snapshot.txt").write_text(content, encoding="utf-8")

    def new_episode(self, idx: int, meta: dict[str, Any]) -> tuple[str, Path]:
        """
        Create an episode directory and register it in manifest.json.

        Returns:
            (episode_id, episode_dir)
        """
        eid = f"{idx:04d}"
        ep_dir = self.ep_root / eid
        ep_dir.mkdir(parents=True, exist_ok=True)

        dump_json(ep_dir / "meta.json", meta)

        self.manifest["episodes"].append(
            {
                "id": eid,
                "road": meta.get("road"),
                "perturbation": meta.get("perturbation"),
                "severity": meta.get("severity"),
                "log": f"episodes/{eid}/log.json",
                "status": "pending",
            }
        )
        dump_json(self.manifest_path, self.manifest)
        return eid, ep_dir

    def complete_episode(self, eid: str, status: str, wall_time_s: float) -> None:
        """
        Update manifest.json with the final episode status and wall time.
        """
        for ep in self.manifest.get("episodes", []):
            if ep.get("id") == eid:
                ep["status"] = status
                ep["wall_time_s"] = round(float(wall_time_s), 3)
                break
        dump_json(self.manifest_path, self.manifest)

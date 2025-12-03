from __future__ import annotations

"""
InterFuser generalization evaluation in CARLA.

- Reads eval/carla/interfuser/cfg_interfuser_generalization.yaml
- For each town entry in `runs`:
    * uses its own routes_files and carla_scenarios_file
    * runs the InterFuser agent via CARLA Leaderboard
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from scripts import abs_path, load_cfg


def run_eval_set(
    carla_cfg: Dict[str, Any],
    agent_cfg: Dict[str, Any],
    results_root: Path,
    run_def: Dict[str, Any],
) -> None:
    run_id = run_def["id"]
    routes_rel_list = run_def["routes_files"]
    scenarios_rel = run_def["carla_scenarios_file"]

    print(f"[eval:carla:interfuser:generalization][INFO] run_id: {run_id}")
    print(f"[eval:carla:interfuser:generalization][INFO] scenarios: {scenarios_rel}")
    print("[eval:carla:interfuser:generalization][INFO] routes_files:")
    for r in routes_rel_list:
        print(f"[eval:carla:interfuser:generalization][INFO] - {r}")

    interfuser_root = abs_path("external/InterFuser")
    sr_dir = interfuser_root / "scenario_runner"
    lb_dir = interfuser_root / "leaderboard"

    launch_script = abs_path(carla_cfg.get("launch_script"))
    agent_script = abs_path(agent_cfg["script"])
    agent_checkpoint = abs_path(agent_cfg["checkpoint"])
    carla_scenarios_file = abs_path(scenarios_rel)

    # CARLA settings
    host = carla_cfg["host"]
    port = str(carla_cfg["port"])
    tm_port = str(carla_cfg.get("traffic_manager_port", 8000))
    tm_seed = str(carla_cfg.get("traffic_manager_seed", 0))
    timeout = str(carla_cfg.get("timeout", "200.0"))
    debug = str(carla_cfg.get("debug", 0))
    track = str(carla_cfg.get("track", "SENSORS"))
    repetitions = str(carla_cfg.get("repetitions", 1))
    weather = str(carla_cfg.get("weather", "none"))
    resume_flag = bool(carla_cfg.get("resume", False))

    for routes_rel in routes_rel_list:
        routes_file = abs_path(routes_rel)
        routes_stem = routes_file.stem

        results_dir = results_root / run_id / routes_stem
        results_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = results_dir / "simulation_results.json"

        print(f"[eval:carla:interfuser:generalization][INFO] routes: {routes_file}")
        print(f"[eval:carla:interfuser:generalization][INFO] results_dir: {results_dir}")
        print(
            f"[eval:carla:interfuser:generalization][INFO] simulation_results: {checkpoint_path}"
        )

        env = os.environ.copy()

        # InterFuser / Leaderboard deps in PYTHONPATH
        extra_path = os.pathsep.join(
            [str(sr_dir), str(lb_dir), str(interfuser_root)]
        )
        env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")

        env["ROUTES"] = str(routes_file)
        env["SCENARIOS"] = str(carla_scenarios_file)
        env["CARLA_HOST"] = host
        env["CARLA_PORT"] = port

        # No PerturbationDrive here
        env["SAVE_PATH"] = str(results_dir)

        cmd: List[str] = [
            sys.executable,
            str(launch_script),
            "--host",
            host,
            "--port",
            port,
            "--trafficManagerPort",
            tm_port,
            "--trafficManagerSeed",
            tm_seed,
            "--timeout",
            timeout,
            "--debug",
            debug,
            "--routes",
            str(routes_file),
            "--scenarios",
            str(carla_scenarios_file),
            "--repetitions",
            repetitions,
            "--weather",
            weather,
            "--track",
            track,
            "--agent",
            str(agent_script),
            "--agent-config",
            str(agent_checkpoint),
            "--checkpoint",
            str(checkpoint_path),
        ]

        if resume_flag:
            cmd.extend(["--resume", "True"])

        print("[eval:carla:interfuser:generalization][INFO] command:")
        print("[eval:carla:interfuser:generalization][INFO] ", " ".join(cmd))

        subprocess.run(cmd, check=True, env=env)


def main() -> int:
    cfg = load_cfg("eval/carla/interfuser/cfg_interfuser_generalization.yaml")

    carla_cfg: Dict[str, Any] = cfg["carla"]
    agent_cfg: Dict[str, Any] = cfg["agent"]
    logging_cfg: Dict[str, Any] = cfg["logging"]

    results_root = abs_path(logging_cfg["runs_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = results_root / ts
    results_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = cfg["runs"]

    for run_def in runs:
        run_eval_set(carla_cfg, agent_cfg, results_root, run_def)

    print("[eval:carla:interfuser:generalization][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

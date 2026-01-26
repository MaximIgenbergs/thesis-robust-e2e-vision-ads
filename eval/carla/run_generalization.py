from __future__ import annotations

"""
CARLA generalization evaluation for multiple leaderboard agents (TCP, InterFuser).

- Reads eval/carla/cfg_generalization.yaml
- Agent/architecture is selected via --model:
    - tcp
    - interfuser
- For each town entry in `runs`:
    - uses its own routes_files and carla_scenarios_file
    - runs the selected agent via its own CARLA Leaderboard repo
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from scripts import abs_path, load_cfg


def run_eval_set(model_name: str, carla_cfg: Dict[str, Any], model_cfg: Dict[str, Any], results_root: Path, run_def: Dict[str, Any]) -> None:
    run_id = run_def["id"]
    routes_rel_list = run_def["routes_files"]
    scenarios_rel = run_def["carla_scenarios_file"]

    print(f"[eval:carla:{model_name}:generalization][INFO] run_id: {run_id}")
    print(f"[eval:carla:{model_name}:generalization][INFO] scenarios: {scenarios_rel}")
    print(f"[eval:carla:{model_name}:generalization][INFO] routes_files:")
    for r in routes_rel_list:
        print(f"[eval:carla:{model_name}:generalization][INFO] - {r}")

    repo_root = abs_path(model_cfg["repo_root"])
    sr_dir = repo_root / "scenario_runner"
    lb_dir = repo_root / "leaderboard"

    launch_script = repo_root / model_cfg["launch_script"]
    agent_script = repo_root / model_cfg["script"]
    agent_checkpoint = abs_path(model_cfg["checkpoint"])
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
        routes_stem = routes_file.stem  # e.g. routes_town02_long

        # logs under: <runs_dir>/<timestamp>/<model>/<TownXX>/<routes_stem>/
        results_dir = results_root / model_name / run_id / routes_stem
        results_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = results_dir / "simulation_results.json"

        print(f"[eval:carla:{model_name}:generalization][INFO] routes: {routes_file}")
        print(f"[eval:carla:{model_name}:generalization][INFO] results_dir: {results_dir}")
        print(f"[eval:carla:{model_name}:generalization][INFO] simulation_results: {checkpoint_path}")

        env = os.environ.copy()

        # Leaderboard / scenario_runner deps in PYTHONPATH
        extra_parts: List[str] = []
        if model_name == "interfuser":
            interfuser_pkg_root = repo_root / "interfuser"
            extra_parts.append(str(interfuser_pkg_root))
        extra_parts.extend([str(sr_dir), str(lb_dir), str(repo_root)])
        extra_path = os.pathsep.join(extra_parts)
        env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")

        env["ROUTES"] = str(routes_file)
        env["SCENARIOS"] = str(carla_scenarios_file)
        env["CARLA_HOST"] = host
        env["CARLA_PORT"] = port
        env["SAVE_PATH"] = str(results_dir)
        
        # Pass save_images configuration to subprocess
        save_images = logging_cfg.get("save_images", False)
        env["SAVE_IMAGES"] = str(save_images).lower()

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
            "--track",
            track,
            "--agent",
            str(agent_script),
            "--agent-config",
            str(agent_checkpoint),
            "--checkpoint",
            str(checkpoint_path),
        ]

        # Only TCP repo / leaderboard supports --weather in your setup
        if model_name == "tcp":
            cmd.extend(["--weather", weather])

        if resume_flag:
            cmd.extend(["--resume", "True"])

        print(f"[eval:carla:{model_name}:generalization][INFO] command:")
        print(f"[eval:carla:{model_name}:generalization][INFO] " + " ".join(cmd))

        subprocess.run(cmd, check=True, env=env)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Select leaderboard agent / architecture to evaluate. Defined under models.* in eval/carla/cfg_generalization.yaml. Examples: --model tcp, --model interfuser")
    args = parser.parse_args()

    cfg = load_cfg("eval/carla/cfg_generalization.yaml")

    carla_cfg: Dict[str, Any] = cfg["carla"]
    models_cfg: Dict[str, Any] = cfg["models"]
    logging_cfg: Dict[str, Any] = cfg["logging"]

    default_model_name = models_cfg.get("default_model", "tcp")
    model_defs = {k: v for k, v in models_cfg.items() if k != "default_model"}

    model_name = args.model or default_model_name
    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not defined under models in eval/carla/cfg_generalization.yaml. Known models: {list(model_defs.keys())}")

    model_cfg = model_defs[model_name]

    results_root = abs_path(logging_cfg["runs_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = results_root / ts
    results_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = cfg["runs"]

    for run_def in runs:
        if not run_def.get("routes_files"):
            print(f"[eval:carla:{model_name}:generalization][INFO] run_id={run_def['id']} has no routes_files, skipping.")
            continue
        run_eval_set(model_name, carla_cfg, model_cfg, results_root, run_def)

    print(f"[eval:carla:{model_name}:generalization][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

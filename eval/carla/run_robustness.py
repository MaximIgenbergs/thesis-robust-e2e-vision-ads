from __future__ import annotations

"""
Robustness evaluation in CARLA using PerturbationDrive.

- Reads eval/carla/cfg_robustness.yaml
- Model/agent is selected via --model:
    - tcp
    - interfuser
- Uses a single routes_file + carla_scenarios_file from `run`
- For each perturbation scenario (incl. optional baseline):
    - runs the selected agent via its CARLA Leaderboard
    - writes results into:
        <runs_dir>/<YYYYmmdd_HHMMSS>/<model>/<scenario_id>/simulation_results.json
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from scripts import abs_path, load_cfg


def build_perturbation_scenarios(pert_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    if pert_cfg.get("baseline", False):
        scenarios.append({"id": "baseline", "route_id": 0, "pd_func": "", "severity": 0})

    severities = pert_cfg.get("severities", [])
    episodes = int(pert_cfg.get("episodes", 1))
    chunks = pert_cfg.get("chunks", [])

    for chunk in chunks:
        for pd_func in chunk:
            for severity in severities:
                for ep in range(episodes):
                    scen_id = f"{pd_func}_s{severity}"
                    if episodes > 1:
                        scen_id = f"{scen_id}_ep{ep}"
                    scenarios.append({"id": scen_id, "route_id": 0, "pd_func": pd_func, "severity": int(severity)})

    return scenarios


def run_scenario(model_name: str, carla_cfg: Dict[str, Any], run_cfg: Dict[str, Any], model_cfg: Dict[str, Any], results_root: Path, scenario: Dict[str, Any]) -> None:
    scen_id = scenario["id"]
    route_id = int(scenario["route_id"])
    pd_func = scenario["pd_func"]
    severity = int(scenario["severity"])
    routes_rel = run_cfg["routes_file"]

    print(f"[eval:carla:{model_name}:robustness][INFO] scenario: {scen_id} route_id: {route_id} perturbation: {pd_func} severity: {severity}")

    repo_root = abs_path(model_cfg["repo_root"])
    sr_dir = repo_root / "scenario_runner"
    lb_dir = repo_root / "leaderboard"

    routes_file = abs_path(routes_rel)
    carla_scenarios_file = abs_path(run_cfg["carla_scenarios_file"])

    scenario_dir = results_root / scen_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = scenario_dir / "simulation_results.json"

    print(f"[eval:carla:{model_name}:robustness][INFO] routes: {routes_file}")
    print(f"[eval:carla:{model_name}:robustness][INFO] scenario_dir: {scenario_dir}")
    print(f"[eval:carla:{model_name}:robustness][INFO] simulation_results: {checkpoint_path}")

    env = os.environ.copy()

    # Leaderboard / scenario_runner deps in PYTHONPATH (+ InterFuser's vendored packages if needed)
    extra_parts: List[str] = []
    if model_name == "interfuser":
        interfuser_pkg_root = repo_root / "interfuser"
        extra_parts.append(str(interfuser_pkg_root))
    extra_parts.extend([str(sr_dir), str(lb_dir), str(repo_root)])
    extra_path = os.pathsep.join(extra_parts)
    env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")

    env["ROUTES"] = str(routes_file)
    env["SCENARIOS"] = str(carla_scenarios_file)

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

    env["CARLA_HOST"] = host
    env["CARLA_PORT"] = port

    env["PD_FUNC"] = pd_func
    env["PD_SEVERITY"] = str(severity)
    env["TCP_PD_FUNC"] = pd_func
    env["TCP_PD_SEVERITY"] = str(severity)

    env["SAVE_PATH"] = str(scenario_dir)
    
    # Pass save_images configuration to subprocess
    save_images = run_cfg.get("save_images", False)
    env["SAVE_IMAGES"] = str(save_images).lower()

    leaderboard_script = repo_root / model_cfg["launch_script"]
    agent_script = abs_path(model_cfg["script"])
    agent_checkpoint = abs_path(model_cfg["checkpoint"])

    cmd: List[str] = [
        sys.executable,
        str(leaderboard_script),
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

    if model_name == "tcp":
        cmd.extend(["--weather", weather])

    if resume_flag:
        cmd.extend(["--resume", "True"])

    print(f"[eval:carla:{model_name}:robustness][INFO] command:")
    print(f"[eval:carla:{model_name}:robustness][INFO] " + " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Select leaderboard agent / architecture to evaluate. Defined under models.* in eval/carla/cfg_robustness.yaml. Examples: --model tcp, --model interfuser")
    args = parser.parse_args()

    cfg = load_cfg("eval/carla/cfg_robustness.yaml")

    carla_cfg: Dict[str, Any] = cfg["carla"]
    run_cfg: Dict[str, Any] = cfg["run"]
    models_cfg: Dict[str, Any] = cfg["models"]
    logging_cfg: Dict[str, Any] = cfg["logging"]
    pert_cfg: Dict[str, Any] = cfg["perturbations"]

    default_model_name = models_cfg.get("default_model", "tcp")
    model_defs = {k: v for k, v in models_cfg.items() if k != "default_model"}

    model_name = args.model or default_model_name
    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not defined under models in eval/carla/cfg_robustness.yaml. Known models: {list(model_defs.keys())}")

    model_cfg = model_defs[model_name]

    results_root = abs_path(logging_cfg["runs_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = results_root / model_name / ts
    results_root.mkdir(parents=True, exist_ok=True)

    scenarios = build_perturbation_scenarios(pert_cfg)

    for scenario in scenarios:
        run_scenario(model_name, carla_cfg, run_cfg, model_cfg, results_root, scenario)

    print(f"[eval:carla:{model_name}:robustness][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

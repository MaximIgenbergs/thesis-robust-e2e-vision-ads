from __future__ import annotations

"""
TCP robustness evaluation in CARLA using PerturbationDrive.

- Reads eval/carla/tcp/cfg_tcp_robustness.yaml
- Uses a single routes_file + carla_scenarios_file from `run`
- For each perturbation scenario (incl. optional baseline):
    * runs the TCP agent via CARLA Leaderboard
    * writes results into:
        <runs_dir>/<YYYYmmdd_HHMMSS>/<scenario_id>/
          └── simulation_results.json
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

    # Optional clean baseline (no perturbations)
    if pert_cfg.get("baseline", False):
        scenarios.append({
            "id": "baseline",
            "route_id": 0,
            "pd_func": "",
            "severity": 0,
        })

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
                    scenarios.append({
                        "id": scen_id,
                        "route_id": 0,
                        "pd_func": pd_func,
                        "severity": int(severity),
                    })

    return scenarios


def run_scenario(carla_cfg: Dict[str, Any], run_cfg: Dict[str, Any], agent_cfg: Dict[str, Any], results_root: Path, scenario: Dict[str, Any]) -> None:
    scen_id = scenario["id"]
    route_id = int(scenario["route_id"])
    pd_func = scenario["pd_func"]
    severity = int(scenario["severity"])
    routes_rel = run_cfg["routes_file"]

    print(f"[eval:carla:tcp:robustness][INFO] scenario: {scen_id} route_id: {route_id} perturbation: {pd_func} severity: {severity}")

    # TCP / Leaderboard deps in PYTHONPATH
    tcp_root = abs_path("external/TCP")
    sr_dir = tcp_root / "scenario_runner"
    lb_dir = tcp_root / "leaderboard"

    routes_file = abs_path(routes_rel)
    carla_scenarios_file = abs_path(run_cfg["carla_scenarios_file"])

    results_dir = results_root
    results_dir.mkdir(parents=True, exist_ok=True)

    scenario_dir = results_dir / scen_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = scenario_dir / "simulation_results.json"

    print(f"[eval:carla:tcp:robustness][INFO] routes: {routes_file}")
    print(f"[eval:carla:tcp:robustness][INFO] results_dir: {results_dir}")
    print(f"[eval:carla:tcp:robustness][INFO] scenario_dir: {scenario_dir}")
    print(f"[eval:carla:tcp:robustness][INFO] simulation_results: {checkpoint_path}")

    env = os.environ.copy()

    # PYTHONPATH
    extra_path = os.pathsep.join([str(sr_dir), str(lb_dir), str(tcp_root)])
    env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")

    # ROUTES / SCENARIOS (absolute, same as generalization)
    env["ROUTES"] = str(routes_file)
    env["SCENARIOS"] = str(carla_scenarios_file)

    # CARLA connection
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

    # PerturbationDrive controls for TCP agent
    env["TCP_PD_FUNC"] = pd_func
    env["TCP_PD_SEVERITY"] = str(severity)

    # Where TCP / Leaderboard writes results
    env["SAVE_PATH"] = str(scenario_dir)

    leaderboard_script = abs_path(carla_cfg.get("launch_script"))

    agent_script = abs_path(agent_cfg["script"])
    agent_checkpoint = abs_path(agent_cfg["checkpoint"])

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

    print("[eval:carla:tcp:robustness][INFO] command:")
    print("[eval:carla:tcp:robustness][INFO] ", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    cfg = load_cfg("eval/carla/tcp/cfg_tcp_robustness.yaml")

    carla_cfg: Dict[str, Any] = cfg["carla"]
    run_cfg: Dict[str, Any] = cfg["run"]
    agent_cfg: Dict[str, Any] = cfg["agent"]
    logging_cfg: Dict[str, Any] = cfg["logging"]
    pert_cfg: Dict[str, Any] = cfg["perturbations"]

    results_root = abs_path(logging_cfg["runs_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = results_root / ts
    results_root.mkdir(parents=True, exist_ok=True)

    scenarios = build_perturbation_scenarios(pert_cfg)

    for scenario in scenarios:
        run_scenario(carla_cfg, run_cfg, agent_cfg, results_root, scenario)

    print("[eval:carla:tcp:robustness][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

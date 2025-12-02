from __future__ import annotations

"""
TCP robustness evaluation in CARLA using PerturbationDrive.
"""

import os
import subprocess
import sys
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
                    scenarios.append({
                        "id": scen_id,
                        "route_id": 0,
                        "pd_func": pd_func,
                        "severity": int(severity),
                    })

    return scenarios


def run_scenario(carla_cfg: Dict[str, Any], run_cfg: Dict[str, Any], agent_cfg: Dict[str, Any], results_dir: Path, scenario: Dict[str, Any]) -> None:
    scen_id = scenario["id"]
    route_id = int(scenario["route_id"])
    pd_func = scenario["pd_func"]
    severity = int(scenario["severity"])

    print(
        f"[eval:carla:tcp:robustness][INFO] "
        f"scenario={scen_id} route_id={route_id} perturbation={pd_func} severity={severity}"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir = results_dir / scen_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    # TCP / Leaderboard deps in PYTHONPATH
    tcp_root = abs_path("external/TCP")
    sr_dir = tcp_root / "scenario_runner"
    lb_dir = tcp_root / "leaderboard"
    extra_path = os.pathsep.join([str(sr_dir), str(lb_dir), str(tcp_root)])
    env["PYTHONPATH"] = extra_path + os.pathsep + env.get("PYTHONPATH", "")
    env["ROUTES"] = run_cfg["routes_file"]
    env["SCENARIOS"] = run_cfg["carla_scenarios_file"]

    # CARLA connection
    env["CARLA_HOST"] = carla_cfg["host"]
    env["CARLA_PORT"] = str(carla_cfg["port"])

    # PerturbationDrive controls for TCP agent
    env["TCP_PD_FUNC"] = pd_func
    env["TCP_PD_SEVERITY"] = str(severity)

    # Where TCP / Leaderboard writes results
    env["SAVE_PATH"] = str(scenario_dir)

    leaderboard_script = abs_path(carla_cfg.get("launch_script"))
    routes_file = abs_path(run_cfg["routes_file"])
    carla_scenarios_file = abs_path(run_cfg["carla_scenarios_file"])

    agent_script = abs_path(agent_cfg["script"])
    agent_checkpoint = abs_path(agent_cfg["checkpoint"])

    cmd = [
        sys.executable,
        str(leaderboard_script),
        "--host", carla_cfg["host"],
        "--port", str(carla_cfg["port"]),
        f"--trafficManagerPort={carla_cfg['traffic_manager_port']}",
        f"--trafficManagerSeed={carla_cfg['traffic_manager_seed']}",
        f"--timeout={carla_cfg['timeout']}",
        "--track", str(carla_cfg["track"]),
        f"--repetitions={carla_cfg['repetitions']}",
        f"--routes={routes_file}",
        f"--scenarios={carla_scenarios_file}",
        f"--agent={agent_script}",
        f"--agent-config={agent_checkpoint}",
        f"--debug={carla_cfg['debug']}",
    ]

    # Optional flags
    if carla_cfg.get("resume", False):
        cmd.append("--resume")

    record_root = str(carla_cfg.get("record_root", "")).strip()
    if record_root:
        cmd.append(f"--record={record_root}")

    print("[eval:carla:tcp:robustness][INFO] command:")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    cfg = load_cfg("eval/carla/tcp/cfg_tcp_robustness.yaml")

    carla_cfg = cfg["carla"]
    run_cfg = cfg["run"]
    agent_cfg = cfg["agent"]
    logging_cfg = cfg["logging"]
    pert_cfg = cfg["perturbations"]

    results_dir = abs_path(logging_cfg["runs_dir"])

    scenarios = build_perturbation_scenarios(pert_cfg)

    for scenario in scenarios:
        run_scenario(carla_cfg, run_cfg, agent_cfg, results_dir, scenario)

    print("[eval:carla:tcp:robustness][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

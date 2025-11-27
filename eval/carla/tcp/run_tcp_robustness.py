from __future__ import annotations

"""
TCP robustness evaluation in CARLA.

- Reads eval/carla/tcp/cfg_tcp_robustness.yaml
- Loads perturbation scenarios from scripts/carla/scenarios/perturbation_scenarios.yaml
- For each scenario, sets TCP_PD_FUNC / TCP_PD_SEVERITY and calls the Leaderboard
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scripts import abs_path


def load_cfg() -> Dict[str, Any]:
    cfg_path = Path(__file__).with_name("cfg_tcp_robustness.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_perturbation_scenarios(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # expect a top-level list of {id, route_id, pd_func, severity}
    return list(data)


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

    # CARLA
    env["CARLA_HOST"] = carla_cfg["host"]
    env["CARLA_PORT"] = str(carla_cfg["port"])

    # PerturbationDrive controls for TCP agent
    env["TCP_PD_FUNC"] = pd_func
    env["TCP_PD_SEVERITY"] = str(severity)

    # Where TCP / Leaderboard writes results
    env["SAVE_PATH"] = str(scenario_dir)

    leaderboard_script = abs_path(run_cfg["script"])
    routes_file = abs_path(run_cfg["routes_file"])
    carla_scenarios_file = abs_path(run_cfg["carla_scenarios_file"])

    agent_script = abs_path(agent_cfg["script"])
    agent_checkpoint = abs_path(agent_cfg["checkpoint"])

    cmd = [
        sys.executable,
        str(leaderboard_script),
        "--host", carla_cfg["host"],
        "--port", str(carla_cfg["port"]),
        "--repetitions=1",
        "--track", "SENSORS",
        f"--routes={routes_file}",
        f"--scenarios={carla_scenarios_file}",
        f"--agent={agent_script}",
        f"--agent-config={agent_checkpoint}",
        "--debug=0",
    ]

    print("[eval:carla:tcp:robustness][INFO] command:")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    cfg = load_cfg()

    carla_cfg = cfg["carla"]
    run_cfg = cfg["run"]
    agent_cfg = cfg["agent"]
    out_cfg = cfg["output"]

    results_dir = abs_path(out_cfg["results_dir"])
    pert_file = abs_path(run_cfg["perturbation_scenarios_file"])
    scenarios = load_perturbation_scenarios(pert_file)

    for scenario in scenarios:
        run_scenario(carla_cfg, run_cfg, agent_cfg, results_dir, scenario)

    print("[eval:carla:tcp:robustness][INFO] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

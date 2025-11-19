import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD = ROOT / "external" / "perturbation-drive"
if str(PD) not in sys.path:
    sys.path.insert(0, str(PD))

import yaml


@dataclass
class ScenarioConfig:
    id: str
    town: str
    route_id: int
    tcp_pd_func: str
    tcp_pd_severity: int


@dataclass
class RunnerConfig:
    carla_host: str
    carla_port: int
    leaderboard_script: Path
    routes_file: Path
    traffic_scenarios_file: Path
    agent_script: Path
    agent_config: Path
    results_dir: Path
    scenarios: List[ScenarioConfig]


def load_config() -> RunnerConfig:
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / "sims" / "carla" / "robustness" / "configs" / "tcp_robustness.yaml"

    with cfg_path.open("r") as f:
        raw = yaml.safe_load(f)

    def p(rel: str) -> Path:
        path = Path(rel)
        if path.is_absolute():
            return path
        return (root / path).resolve()
    
    scenarios = [
        ScenarioConfig(
            id=s["id"],
            town=s["town"],
            route_id=int(s["route_id"]),
            tcp_pd_func=s.get("tcp_pd_func", ""),
            tcp_pd_severity=int(s.get("tcp_pd_severity", 0)),
        )
        for s in raw["scenarios"]
    ]

    return RunnerConfig(
        carla_host=raw.get("carla_host", "localhost"),
        carla_port=int(raw.get("carla_port", 2000)),
        leaderboard_script=p(raw["leaderboard_script"]),
        routes_file=p(raw["routes_file"]),
        traffic_scenarios_file=p(raw["traffic_scenarios_file"]),
        agent_script=p(raw["agent_script"]),
        agent_config=p(raw["agent_config"]),
        results_dir=p(raw["results_dir"]),
        scenarios=scenarios,
    )


def run_scenario(cfg: RunnerConfig, sc: ScenarioConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = cfg.results_dir / sc.id
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[3]
    tcp_root = root / "external" / "TCP"

    env = os.environ.copy()

    # --- Correct PYTHONPATH handling ---
    sr_dir = tcp_root / "scenario_runner"
    lb_dir = tcp_root / "leaderboard"

    old_pp = env.get("PYTHONPATH", "")
    extra_paths = os.pathsep.join(str(p) for p in [sr_dir, lb_dir, tcp_root])

    env["PYTHONPATH"] = extra_paths if not old_pp else extra_paths + os.pathsep + old_pp

    # --- Carla / PD settings ---
    env["CARLA_HOST"] = cfg.carla_host
    env["CARLA_PORT"] = str(cfg.carla_port)

    env["TCP_PD_FUNC"] = sc.tcp_pd_func or ""
    env["TCP_PD_SEVERITY"] = str(sc.tcp_pd_severity)
    env["PD_SCENARIO_ID"] = sc.id

    # --- Command (NO --route-id) ---
    cmd = [
        sys.executable,
        str(cfg.leaderboard_script),
        "--host", cfg.carla_host,
        "--port", str(cfg.carla_port),
        "--repetitions=1",
        "--track", "SENSORS",
        f"--routes={cfg.routes_file}",
        f"--scenarios={cfg.traffic_scenarios_file}",
        f"--agent={cfg.agent_script}",
        f"--agent-config={cfg.agent_config}",
        "--debug=0",
    ]

    print(f"[run_tcp_robustness] running scenario {sc.id}")
    subprocess.run(cmd, env=env, check=True, cwd=tcp_root)


def main() -> None:
    cfg = load_config()
    for sc in cfg.scenarios:
        run_scenario(cfg, sc)


if __name__ == "__main__":
    main()

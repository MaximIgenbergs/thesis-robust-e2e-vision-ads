from __future__ import annotations

"""
Run CARLA robustness evaluation over a list of PerturbationDrive scenarios.

This script:
  - reads scenario definitions from configs/perturbation_scenarios.py
  - for each scenario:
      * picks the correct town + route_id (routes_lav_valuid.xml)
      * optionally inspects the weather from XML (for logging)
      * calls the Leaderboard evaluator with a PD perturbation config
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.carla.robustness.configs.perturbation_scenarios import (
    SCENARIOS,
    get_scenario_by_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CARLA robustness evaluation for PerturbationDrive scenarios."
    )

    parser.add_argument(
        "--routes-file",
        type=str,
        default="leaderboard/data/evaluation_routes/routes_lav_valuid.xml",
        help="Path to CARLA routes XML used by the Leaderboard.",
    )
    parser.add_argument(
        "--traffic-scenarios-file",
        type=str,
        default="leaderboard/data/scenarios/all_towns_traffic_scenarios.json",
        help="Path to all_towns_traffic_scenarios.json.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="tcp",
        help="Agent name to filter scenarios by (e.g. 'tcp', 'interfuser').",
    )
    parser.add_argument(
        "--scenario-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of scenario IDs to run. "
             "If omitted, all scenarios for the given agent are executed.",
    )
    parser.add_argument(
        "--leaderboard-script",
        type=str,
        default="leaderboard/leaderboard/leaderboard_evaluator.py",
        help="Entry point of the CARLA Leaderboard evaluator.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/carla_pd_scenarios",
        help="Output folder for logs/metrics.",
    )

    return parser.parse_args()


def load_weather_for_route(
    routes_file: str,
    town: str,
    route_id: int,
) -> Optional[Dict[str, str]]:
    """
    Parse the routes XML to fetch the <weather ...> attributes for a given
    (town, route_id) pair. This does not change CARLA's behavior; it is purely
    for logging/metadata.
    """
    tree = ET.parse(routes_file)
    root = tree.getroot()

    for route in root.findall("route"):
        rid = int(route.get("id", "-1"))
        rtown = route.get("town", "")
        if rid != route_id or rtown != town:
            continue

        weather = route.find("weather")
        if weather is None:
            return None

        return dict(weather.attrib)

    return None


def select_scenarios(
    agent: str,
    scenario_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Filter SCENARIOS by agent and optional list of IDs.
    """
    candidates = [s for s in SCENARIOS if s.get("agent") == agent]

    if scenario_ids is None:
        return candidates

    selected = []
    wanted = set(scenario_ids)
    for sid in wanted:
        selected.append(get_scenario_by_id(sid))

    return selected


def build_env_for_scenario(
    base_env: Dict[str, str],
    scenario: Dict[str, Any],
) -> Dict[str, str]:
    """
    Attach the PD perturbation config to the environment so your CARLA
    wrapper / PerturbationDrive integration can read it.

    You will need to consume PD_PERTURBATIONS in your camera or evaluator code.
    """
    env = base_env.copy()
    env["PD_SCENARIO_ID"] = scenario["id"]
    env["PD_TOWN"] = scenario["town"]
    env["PD_ROUTE_ID"] = str(scenario["route_id"])
    env["PD_PERTURBATIONS"] = json.dumps(scenario["perturbations"])
    return env


def run_single_scenario(
    args: argparse.Namespace,
    scenario: Dict[str, Any],
) -> None:
    town = scenario["town"]
    route_id = scenario["route_id"]

    weather = load_weather_for_route(args.routes_file, town, route_id)

    print()
    print("=" * 80)
    print(f"Running PD scenario: {scenario['id']}")
    print(f"  town       : {town}")
    print(f"  route_id   : {route_id}")
    print(f"  agent      : {scenario.get('agent')}")
    print(f"  group      : {scenario.get('group')}")
    print(f"  desc       : {scenario.get('description')}")
    if weather:
        print(f"  weather    : {weather.get('id')} "
              f"(cloudiness={weather.get('cloudiness')}, "
              f"precip={weather.get('precipitation')}, "
              f"sun_alt={weather.get('sun_altitude_angle')})")

    # Make sure results dir exists
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Example: per-scenario log dir (you can wire this into your evaluator)
    scenario_log_dir = results_dir / scenario["id"]
    scenario_log_dir.mkdir(exist_ok=True)

    # Build environment with PD configuration
    base_env = os.environ.copy()
    env = build_env_for_scenario(base_env, scenario)
    env["PD_OUTPUT_DIR"] = str(scenario_log_dir)

    # Call the Leaderboard evaluator for this (town, route_id).
    #
    # You might already have a wrapper script instead of calling the evaluator
    # directly. If so, replace this with your existing call-site, keeping
    # the PD_* env variables intact.
    cmd = [
        sys.executable,
        args.leaderboard_script,
        f"--routes={args.routes_file}",
        f"--scenarios={args.traffic_scenarios_file}",
        f"--route-id={route_id}",
        # Add your usual flags here, e.g.:
        # f"--agent={path_to_your_agent}",
        # f"--agent-config={path_to_agent_config}",
        # f"--debug={0 or 1}",
    ]

    print("  command    :", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    args = parse_args()

    scenarios = select_scenarios(
        agent=args.agent,
        scenario_ids=args.scenario_ids,
    )

    if not scenarios:
        print(f"No scenarios selected for agent={args.agent}.")
        return

    print(f"Selected {len(scenarios)} scenario(s) for agent '{args.agent}'.")
    for s in scenarios:
        print(f"  - {s['id']} (town={s['town']}, route_id={s['route_id']})")

    for scenario in scenarios:
        run_single_scenario(args, scenario)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
PerturbationDrive robustness scenarios for CARLA Leaderboard-style evaluation.

Each scenario combines:
  - a CARLA town and route_id (from routes_*.xml)
  - an agent name ("tcp", "interfuser", ...)
  - a list of perturbations to apply in PerturbationDrive

We do NOT change the original Leaderboard route or scenario files.
We only attach perturbations on top.

Route IDs and towns correspond to:
  leaderboard/data/evaluation_routes/routes_lav_valuid.xml

Traffic scenarios (NPCs etc.) come from:
  leaderboard/data/scenarios/all_towns_traffic_scenarios.json
"""

from typing import Any, Dict, List


ScenarioDict = Dict[str, Any]


SCENARIOS: List[ScenarioDict] = [
    # --- Town02, long loop, different weathers (routes 0–3) ---

    {
        "id": "t02_r0_clearnoon_baseline_gaussian_blur",
        "town": "Town02",
        "route_id": 0,
        "agent": "tcp",
        "group": "camera_baseline",
        "description": "Town02 long loop, ClearNoon, mild Gaussian blur on front camera.",
        "perturbations": [
            {
                "op": "camera_gaussian_blur",
                "target": "rgb_front",
                "kwargs": {
                    "sigma": 1.0
                },
            }
        ],
    },
    {
        "id": "t02_r1_cloudysunset_strong_blur",
        "town": "Town02",
        "route_id": 1,
        "agent": "tcp",
        "group": "camera_weather_sweep",
        "description": "Town02 long loop, CloudySunset, stronger Gaussian blur.",
        "perturbations": [
            {
                "op": "camera_gaussian_blur",
                "target": "rgb_front",
                "kwargs": {
                    "sigma": 2.0
                },
            }
        ],
    },
    {
        "id": "t02_r2_softrain_glare",
        "town": "Town02",
        "route_id": 2,
        "agent": "tcp",
        "group": "lighting",
        "description": "Town02, SoftRainDawn, increased brightness and reduced contrast.",
        "perturbations": [
            {
                "op": "camera_gamma",
                "target": "rgb_front",
                "kwargs": {
                    "gamma": 0.8,
                    "gain": 1.2
                },
            }
        ],
    },
    {
        "id": "t02_r3_hardrainnight_noise",
        "town": "Town02",
        "route_id": 3,
        "agent": "tcp",
        "group": "noise",
        "description": "Town02, HardRainNight, additive Gaussian sensor noise.",
        "perturbations": [
            {
                "op": "camera_additive_gaussian_noise",
                "target": "rgb_front",
                "kwargs": {
                    "std": 0.05
                },
            }
        ],
    },

    # --- Town05, highway-ish routes (8–15) ---

    {
        "id": "t05_r8_clearnoon_random_occlusions",
        "town": "Town05",
        "route_id": 8,
        "agent": "interfuser",
        "group": "occlusion",
        "description": "Town05, ClearNoon, random rectangular occlusions on front camera.",
        "perturbations": [
            {
                "op": "camera_random_box_occlusion",
                "target": "rgb_front",
                "kwargs": {
                    "num_boxes": 3,
                    "box_size_min": 0.05,
                    "box_size_max": 0.15,
                },
            }
        ],
    },
    {
        "id": "t05_r9_cloudysunset_rain_drops",
        "town": "Town05",
        "route_id": 9,
        "agent": "interfuser",
        "group": "weather_appearance",
        "description": "Town05, CloudySunset, synthetic raindrop streaks on the lens.",
        "perturbations": [
            {
                "op": "camera_rain_drops",
                "target": "rgb_front",
                "kwargs": {
                    "density": 0.4,
                    "streak_length": 0.6,
                },
            }
        ],
    },
    {
        "id": "t05_r10_softrain_color_shift",
        "town": "Town05",
        "route_id": 10,
        "agent": "interfuser",
        "group": "color",
        "description": "Town05, SoftRainDawn, small hue and saturation shift.",
        "perturbations": [
            {
                "op": "camera_hsv_jitter",
                "target": "rgb_front",
                "kwargs": {
                    "max_hue_shift": 0.05,
                    "max_saturation_scale": 0.15,
                },
            }
        ],
    },
    {
        "id": "t05_r11_hardrainnight_combo",
        "town": "Town05",
        "route_id": 11,
        "agent": "interfuser",
        "group": "combined",
        "description": "Town05, HardRainNight, combined blur + noise + low contrast.",
        "perturbations": [
            {
                "op": "camera_gaussian_blur",
                "target": "rgb_front",
                "kwargs": {"sigma": 1.0},
            },
            {
                "op": "camera_additive_gaussian_noise",
                "target": "rgb_front",
                "kwargs": {"std": 0.03},
            },
            {
                "op": "camera_gamma",
                "target": "rgb_front",
                "kwargs": {"gamma": 1.2, "gain": 0.9},
            },
        ],
    },

    # You can keep adding:
    # - more Town02/Town05 routes (route_id 4–7, 12–15)
    # - variants for different agents
    # - “stress test” bundles with several perturbations
]


def get_scenario_by_id(scenario_id: str) -> ScenarioDict:
    for scenario in SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    raise KeyError(f"Unknown scenario id: {scenario_id}")


def list_scenarios(agent: str | None = None) -> List[ScenarioDict]:
    if agent is None:
        return list(SCENARIOS)
    return [s for s in SCENARIOS if s.get("agent") == agent]

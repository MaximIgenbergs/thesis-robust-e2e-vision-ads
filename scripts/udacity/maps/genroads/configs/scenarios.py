from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List


class Control(str, Enum):
    STEERING = "steering"
    THROTTLE = "throttle"


class Mode(str, Enum):
    MUL = "mul" # multiply base value
    ADD = "add" # add delta
    SET = "set" # hard override


@dataclass
class Perturbation:
    control: Control # steering, throttle
    mode: Mode # mul, add, set
    factor: float # meaning depends on mode
    start_wp: int # inclusive
    end_wp: int # inclusive
    comment: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["control"] = self.control.value
        d["mode"] = self.mode.value
        return d


@dataclass
class Scenario:
    name: str # unique id, goes into episode meta
    active: bool # whether to run this scenario
    road: str # key in roads.ROADS
    level: int # difficulty level (1, 2, 3, ...)
    perturbations: List[Perturbation]
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "road": self.road,
            "level": self.level,
            "comment": self.comment,
            "perturbations": [p.to_dict() for p in self.perturbations],
        }


def scenario_bounds(scenario: Scenario) -> tuple[int, int]:
    """Global [start, end] waypoint index range covered by this scenario."""
    starts = [p.start_wp for p in scenario.perturbations]
    ends = [p.end_wp for p in scenario.perturbations]
    return min(starts), max(ends)

# Scenarios organized by road name
SCENARIOS: Dict[str, List[Scenario]] = {
    "c": [
        Scenario(
            name="fullspeed before turn",
            active=False,
            road="c",
            level=1,
            comment="Max acceleration before turn 1",
            perturbations=[
                Perturbation(
                    control=Control.THROTTLE,
                    mode=Mode.SET,
                    factor=1,
                    start_wp=4,
                    end_wp=30,
                    comment="Speeds up into the curve until WP 30.",
                ),
            ],
        ),
        Scenario(
            name="fullspeed before turn",
            active=False,
            road="c",
            level=2,
            comment="Max acceleration before turn 1",
            perturbations=[
                Perturbation(
                    control=Control.THROTTLE,
                    mode=Mode.SET,
                    factor=1.0,
                    start_wp=4,
                    end_wp=50,
                    comment="Speeds up into the curve until WP 50.",
                ),
            ],
        ),
        Scenario(
            name="Accelerate and kick steering",
            active=True,
            road="c",
            level=1,
            comment="Short steering kick before turnafter accelerating",
            perturbations=[
                Perturbation(
                    control=Control.STEERING,
                    mode=Mode.SET,
                    factor=+0.6, # kick to the right
                    start_wp=90,
                    end_wp=95,
                    comment="Short steering kick to the right.",
                ),
                Perturbation(
                    control=Control.THROTTLE,
                    mode=Mode.SET,
                    factor=1.0, # full throttle
                    start_wp=4,
                    end_wp=20,
                    comment="Speeds up into the curve until WP 20.",
                ),
            ],
        ),
    ],
}

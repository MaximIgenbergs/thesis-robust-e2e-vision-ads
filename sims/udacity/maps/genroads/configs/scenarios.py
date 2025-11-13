# sims/udacity/maps/genroads/configs/scenarios.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List


class Control(str, Enum):
    STEERING = "steering"
    THROTTLE = "throttle"


class Mode(str, Enum):
    MUL = "mul"   # multiply base value
    ADD = "add"   # add delta
    SET = "set"   # hard override


@dataclass
class Perturbation:
    control: Control          # steering or throttle
    mode: Mode                # mul/add/set
    factor: float             # meaning depends on mode
    start_wp: int             # inclusive waypoint index
    end_wp: int               # inclusive waypoint index
    comment: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["control"] = self.control.value
        d["mode"] = self.mode.value
        return d


@dataclass
class Scenario:
    name: str                 # unique id, goes into episode meta
    road: str                 # key in roads.ROADS
    level: int                # difficulty level (1, 2, 3, ...)
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


SCENARIOS_BY_ROAD: Dict[str, List[Scenario]] = {
    "c": [
        Scenario(
            name="c_understeer_and_fast_lvl1",
            road="c",
            level=1,
            comment="Slight understeer in the curve + mild overspeed before it.",
            perturbations=[
                Perturbation(
                    control=Control.STEERING,
                    mode=Mode.MUL,
                    factor=0.7,          # reduce steering magnitude
                    start_wp=55,
                    end_wp=75,
                    comment="Understeer through main curve.",
                ),
                Perturbation(
                    control=Control.THROTTLE,
                    mode=Mode.MUL,
                    factor=1.2,          # a bit too fast
                    start_wp=45,
                    end_wp=60,
                    comment="Speed up into the curve.",
                ),
            ],
        ),
        Scenario(
            name="c_steer_kick_and_brake_lvl2",
            road="c",
            level=2,
            comment="Kick steering on straight, then brief braking.",
            perturbations=[
                Perturbation(
                    control=Control.STEERING,
                    mode=Mode.ADD,
                    factor=+0.20,        # kick to the right
                    start_wp=30,
                    end_wp=36,
                    comment="Short steering kick on straight.",
                ),
                Perturbation(
                    control=Control.THROTTLE,
                    mode=Mode.SET,
                    factor=-0.5,         # braking (throttle in [-1, 1])
                    start_wp=37,
                    end_wp=45,
                    comment="Brief braking after the kick.",
                ),
            ],
        ),
    ],
}

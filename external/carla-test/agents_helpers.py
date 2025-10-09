# -*- coding: utf-8 -*-
"""
agents_helpers.py — navigation/agent helpers for CARLA.

Provides:
- build_grp(world, sampling_resolution)
- pick_routes(world, grp, n, min_m, max_m, rng)
- next_high_level_command(route, idx, lookahead=60)
- make_agent(world, ego, behavior='cautious')
"""

from __future__ import annotations
from typing import List, Tuple

import carla  # type: ignore
from carla.agents.navigation.behavior_agent import BehaviorAgent
from carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from carla.agents.navigation.local_planner import RoadOption

RouteType = List[Tuple[carla.Waypoint, RoadOption]]

__all__ = ["build_grp", "pick_routes", "next_high_level_command", "make_agent"]


def build_grp(world: "carla.World", sampling_resolution: float = 2.0) -> GlobalRoutePlanner:
    return GlobalRoutePlanner(world.get_map(), sampling_resolution)


def _route_distance(route: RouteType) -> float:
    d = 0.0
    for i in range(1, len(route)):
        a = route[i - 1][0].transform.location
        b = route[i][0].transform.location
        d += a.distance(b)
    return d


def pick_routes(world: "carla.World",
                grp: GlobalRoutePlanner,
                n: int,
                min_m: float,
                max_m: float,
                rng) -> List[RouteType]:
    sp = world.get_map().get_spawn_points()
    sp = sorted(sp, key=lambda t: (t.location.x, t.location.y, t.location.z))
    routes: List[RouteType] = []
    tries, max_tries = 0, 2000
    while len(routes) < n and tries < max_tries:
        tries += 1
        s = rng.randrange(len(sp)); e = rng.randrange(len(sp))
        if s == e: continue
        r = grp.trace_route(sp[s].location, sp[e].location)
        if min_m <= _route_distance(r) <= max_m:
            routes.append(r)
    return routes


def next_high_level_command(route: RouteType, idx: int, lookahead: int = 60) -> str:
    upper = min(idx + lookahead, len(route))
    for j in range(idx, upper):
        opt = route[j][1]
        if opt in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT):
            return opt.name
    return RoadOption.LANEFOLLOW.name


def make_agent(world: "carla.World", ego: "carla.Vehicle", behavior: str = "cautious") -> BehaviorAgent:
    return BehaviorAgent(ego, behavior=behavior)

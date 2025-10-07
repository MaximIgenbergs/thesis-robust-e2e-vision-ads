import carla
from carla.agents.navigation.behavior_agent import BehaviorAgent
from carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from carla.agents.navigation.local_planner import RoadOption
from carla.agents.navigation.local_planner import RoadOption

def build_grp(world, sampling_resolution=2.0):
    return GlobalRoutePlanner(world.get_map(), sampling_resolution)

def route_distance(route):
    d = 0.0
    for i in range(1, len(route)):
        a, b = route[i-1][0].transform.location, route[i][0].transform.location
        d += a.distance(b)
    return d

def pick_routes(world, grp, n, min_m, max_m, rng):
    sp = world.get_map().get_spawn_points()
    sp = sorted(sp, key=lambda t:(t.location.x, t.location.y, t.location.z))
    routes = []
    tries = 0
    while len(routes) < n and tries < 2000:
        tries += 1
        start = rng.randrange(len(sp))
        end   = rng.randrange(len(sp))
        if start == end: continue
        r = grp.trace_route(sp[start].location, sp[end].location)
        dist = route_distance(r)
        if min_m <= dist <= max_m:
            routes.append(r)
    return routes

def next_high_level_command(route, idx):
    # look ahead in the route for the next maneuver that is not LANEFOLLOW
    for j in range(idx, min(idx+60, len(route))):
        opt = route[j][1]
        if opt in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT):
            return opt.name
    return RoadOption.LANEFOLLOW.name

def make_agent(world, ego):
    agent = BehaviorAgent(ego, behavior='cautious')  # 'cautious'/'normal'/'aggressive'
    return agent

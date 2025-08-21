import udacity_gym.global_manager as _gm
_gm.get_simulator_state = lambda: {}

import multiprocessing as mp
mp.set_start_method("fork", force=True)

import sys
import pathlib

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
from udacity_gym.action import UdacityAction
from udacity_gym.agent import PIDUdacityAgent_Angle
from udacity_gym.agent_callback import LogObservationCallback
from models.utils.utils import make_collection_dir
from utils.conf import (
    simulator_infos,
    TRACK,
    DAYTIME,
    WEATHER,
    ENABLE_LOGGING,
    STEPS,
    EGO_CIRCUIT_NAME,
    EGO_SPAWN_INDEX,
    FIXED_NPCS,
    RANDOM_NPCS,
)

if __name__ == '__main__':
    if ENABLE_LOGGING:
        log_directory = make_collection_dir("pid")
        print(f"Logging to {log_directory}")
    else:
        print("Logging is disabled.")

    sim_info = simulator_infos[1]
    assert pathlib.Path(sim_info['exe_path']).exists(), f"Simulator binary not found at {sim_info['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=sim_info['exe_path'],
        host=sim_info['host'],
        cmd_port=sim_info['cmd_port'],
        telemetry_port=sim_info['telemetry_port'],
        event_port=sim_info['event_port'],
        others_port=sim_info['others_port']
    )
    env = UdacityGym(simulator=simulator)

    npc_cars = []
    for npc in FIXED_NPCS:
        npc_cars.append({
            "name": npc.get("name", "NPC"),
            "prefab_name": npc.get("prefab", "Objects/CarRed"),
            "autonomous": False,
            "speed": npc.get("speed", 25.0),
            "layer": "Road",
            "waypoints": [npc["circuit"]],
            "spawn_point": float(npc["spawn_index"]),
            "offset": [0.0, 0.0, 0.0],
            "scale_Vektor": [1.0, 1.0, 1.0],
            "rotation": [0.0, 0.0, 0.0],
            "humanBehavior": 0.0,
            "waitingPoints": []
        })

    observation, _ = simulator.reset(
        new_track_name=TRACK,
        new_weather_name=WEATHER,
        new_daytime_name=DAYTIME,
        ego_circuit_name=EGO_CIRCUIT_NAME,
        ego_spawn_index=int(EGO_SPAWN_INDEX),
        npc_cars=npc_cars,
        random_npc_cars=RANDOM_NPCS,
    )

    log_cb = LogObservationCallback(log_directory) if ENABLE_LOGGING else None

    agent = PIDUdacityAgent_Angle(
        track=TRACK,
        before_action_callbacks=[],
        after_action_callbacks=[log_cb] if ENABLE_LOGGING else []
    )

    info = None
    try:
        for _ in tqdm.tqdm(range(STEPS)):
            action = agent(observation)
            last_obs = observation
            observation, reward, terminated, truncated, info = env.step(action)
            while observation.time == last_obs.time:
                time.sleep(0.0025)
                observation = env.observe()
    except KeyboardInterrupt:
        print("Execution interrupted by user. Saving logs and exiting...")
    finally:
        if ENABLE_LOGGING and info:
            with open(log_directory / "info.json", "w") as f:
                json.dump(info, f)
        if ENABLE_LOGGING and log_cb and log_cb.logs:
            log_cb.save()
            print(f"Logs saved to {log_directory}")
        else:
            print("No observations were recorded. Nothing to save.")
        simulator.close()
        env.close()
        print("Experiment concluded.")

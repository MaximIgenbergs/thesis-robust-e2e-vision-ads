import udacity_gym.global_manager as _gm
_gm.get_simulator_state = lambda: {}  # Fix for macOS spawn issue

import multiprocessing as mp
mp.set_start_method("fork", force=True)

import sys
import pathlib

# Add project root to PYTHONPATH so shared utils can be imported
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
from udacity_gym.action import UdacityAction
from udacity_gym.agent import PIDUdacityAgent_Angle
from udacity_gym.agent_callback import LogObservationCallback
from utils.conf import Track_Infos
from models.utils.utils import make_collection_dir

# Configuration
track_index = 2  # jungle
logging = True
steps = 7000

if __name__ == '__main__':
    # Track & simulator settings
    track_info = Track_Infos[track_index]
    track      = track_info['track_name']
    sim_info   = track_info['simulator']
    daytime    = 'day'
    weather    = 'sunny'

    if logging:
        log_directory = make_collection_dir('pid')
        print(f"Logging to {log_directory}")
    else:
        print("Logging is disabled.")

    # Initialize simulator & environment
    assert pathlib.Path(sim_info['exe_path']).exists(), \
        f"Simulator binary not found at {sim_info['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=sim_info['exe_path'],
        host=sim_info['host'],
        cmd_port=sim_info['cmd_port'],
        telemetry_port=sim_info['telemetry_port'],
        event_port=sim_info['event_port']
    )
    env = UdacityGym(simulator=simulator)

    # Reset and wait for readiness
    observation, _ = env.reset(track=track, weather=weather, daytime=daytime)
    while not observation or not observation.is_ready():
        print("Waiting for environment to set up...")
        time.sleep(1)
        observation = env.observe()

    # Logging callback
    if logging:
        log_cb = LogObservationCallback(log_directory)

    # Instantiate agent
    agent = PIDUdacityAgent_Angle(
        track=track,
        before_action_callbacks=[],
        after_action_callbacks=[log_cb] if logging else []
    )

    # Main loop
    info = None
    try:
        for _ in tqdm.tqdm(range(steps)):
            action = agent(observation)
            last_obs = observation
            observation, reward, terminated, truncated, info = env.step(action)
            # wait for next frame
            while observation.time == last_obs.time:
                time.sleep(0.0025)
                observation = env.observe()
    except KeyboardInterrupt:
        print("Execution interrupted by user. Saving logs and exiting...")
    finally:
        if logging and info:
            with open(log_directory / "info.json", "w") as f:
                json.dump(info, f)
        # Save logs
        if logging and log_cb.logs:
            log_cb.save()
            print(f"Logs saved to {log_directory}")
        else:
            print("No observations were recorded → nothing to save.")
        simulator.close()
        env.close()
        print("Experiment concluded.")

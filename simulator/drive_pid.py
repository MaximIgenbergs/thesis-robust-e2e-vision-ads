import sys
import pathlib

# Add project root to PYTHONPATH so shared utils can be imported
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import json
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym
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

    # Create a new collection directory under data/collections
    log_directory = make_collection_dir('pid')
    print(f"Logging to {log_directory}")

    # Initialize simulator & environment
    assert pathlib.Path(sim_info['exe_path']).exists(), f"Simulator binary not found at {sim_info['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=sim_info['exe_path'], host=sim_info['host'], port=sim_info['port']
    )
    env = UdacityGym(simulator=simulator)
    simulator.start = simulator.sim_executor.start
    simulator.start()

    # Reset and wait for readiness
    observation, _ = env.reset(track=track, weather=weather, daytime=daytime)
    while not observation or not observation.is_ready():
        print("Waiting for environment to set up...")
        time.sleep(1)
        observation = env.observe()

    # Logging callback
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
            while observation.time == last_obs.time:
                time.sleep(0.0025)
                observation = env.observe()
    except KeyboardInterrupt:
        print("Execution interrupted by user. Saving logs and exiting...")
    finally:
        # Save simulator info
        if info:
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

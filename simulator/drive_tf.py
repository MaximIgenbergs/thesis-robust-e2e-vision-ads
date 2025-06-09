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
import numpy as np
import tensorflow as tf
from pathlib import Path
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import UdacityAgent
from udacity_gym.agent_callback import LogObservationCallback, PauseSimulationCallback, ResumeSimulationCallback
from utils.conf import Track_Infos
from models.utils.utils import make_collection_dir
from tensorflow.keras.models import load_model  # type: ignore

# Configuration
collector = 'dave2'
track_index = 2  # jungle
logging = False
steps = 4000
model_path = Path(__file__).resolve().parents[1] / 'models/dave2/models/final_model.h5'

class BaselineCNNAgent(UdacityAgent):
    """
    Very thin wrapper: loads your two-head CNN with compile=False,
    resizes each incoming frame to 66x200, normalizes to [-1,1],
    calls predict(), then clamps steering/throttle.
    """
    def __init__(self, model_path, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        assert model_path.exists(), f"Model file not found: {model_path}"
        self.model = load_model(str(model_path), compile=False)

    def action(self, observation, *args, **kwargs) -> UdacityAction:
        img = observation.input_image
        arr = np.array(img, dtype=np.float32)
        img_tf = tf.image.resize(arr, [66, 200])
        img_tf = img_tf / 127.5 - 1.0
        inp  = tf.expand_dims(img_tf, axis=0)
        pred = self.model.predict(inp, verbose=0)[0]
        steer, thr = float(pred[0]), float(pred[1])
        steer = np.clip(steer, -1.0, 1.0)
        thr   = np.clip(thr,   0.0, 1.0)
        return UdacityAction(steering_angle=steer, throttle=thr)

if __name__ == '__main__':
    # Track & simulator settings
    track_info = Track_Infos[track_index]
    track      = track_info['track_name']
    sim_info   = track_info['simulator']
    daytime    = 'day'
    weather    = 'sunny'

    # Create a new collection directory under data/collections
    log_directory = make_collection_dir(collector)
    print(f"Logging to {log_directory}")

    # Initialize simulator & environment
    assert Path(sim_info['exe_path']).exists(), f"Simulator binary not found at {sim_info['exe_path']}"
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
    agent = BaselineCNNAgent(
        model_path=model_path,
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


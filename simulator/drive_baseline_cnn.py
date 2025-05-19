import udacity_gym.global_manager as _gm
_gm.get_simulator_state = lambda: {}
import json
import pathlib
import time
import tqdm
import numpy as np
import tensorflow as tf
from pathlib import Path
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import UdacityAgent
from udacity_gym.agent_callback import LogObservationCallback, PauseSimulationCallback, ResumeSimulationCallback
from utils.conf import Track_Infos
from tensorflow.keras.models import load_model  # type: ignore

# Configuration
track_index = 2 # jungle
logging = True
steps = 4000  
model_path = Path(__file__).resolve().parents[1] / 'models/baseline_cnn/models/best_model.h5'

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
        # 1) grab raw image
        img = observation.input_image  # PIL.Image or numpy array
        arr = np.array(img, dtype=np.float32)   # likely (80,160,3)

        # 2) resize to (66,200)
        img_tf = tf.image.resize(arr, [66, 200])

        # 3) normalize to [-1,1]
        img_tf = img_tf / 127.5 - 1.0

        # 4) batch & predict [steer, throttle]
        inp  = tf.expand_dims(img_tf, axis=0)    # (1,66,200,3)
        pred = self.model.predict(inp, verbose=0)[0]
        steer, thr = float(pred[0]), float(pred[1])

        # 5) clamp
        steer = np.clip(steer, -1.0, 1.0)
        thr   = np.clip(thr,   0.0, 1.0)

        return UdacityAction(steering_angle=steer, throttle=thr)


if __name__ == '__main__':

    # Simulator connection settings (unused but in PID script)
    host = "127.0.0.1"
    port = 4567

    # Track settings
    track_info   = Track_Infos[track_index]
    track        = track_info['track_name']
    sim_info     = track_info['simulator']
    daytime      = "day"
    weather      = "sunny"
    log_directory = pathlib.Path(f"udacity_dataset_lake_dave/{track}_{weather}_{daytime}_baseline")
    log_directory.mkdir(parents=True, exist_ok=True)
    print(sim_info)

    # Create simulator & gym
    assert pathlib.Path(sim_info['exe_path']).exists(), f"Simulator binary not found at {sim_info['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=sim_info['exe_path'],
        host=sim_info['host'],
        port=sim_info['port'],
    )
    env = UdacityGym(simulator=simulator)

    # Use Unity’s built-in SocketIO server only
    simulator.start = simulator.sim_executor.start
    simulator.start()

    # Reset environment
    observation, _ = env.reset(track=track, weather=weather, daytime=daytime)

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    # Logging callback
    log_cb = LogObservationCallback(log_directory)

    # Instantiate agent
    agent = BaselineCNNAgent(
        model_path=model_path,
        before_action_callbacks=[],
        after_action_callbacks=[log_cb] if logging else None
    )

    # Main loop
    info = None
    try:
        for _ in tqdm.tqdm(range(steps)):
            action = agent(observation)
            last_obs = observation
            observation, reward, terminated, truncated, info = env.step(action)

            # Wait for next frame
            while observation.time == last_obs.time:
                observation = env.observe()
                time.sleep(0.0025)

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

        # Clean shutdown
        simulator.close()
        env.close()
        print("Experiment concluded.")

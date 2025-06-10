import udacity_gym.global_manager as _gm
_gm.get_simulator_state = lambda: {}  # Fix by ChatGPT to remove multiprocessing.Manager spawn on macOS

import multiprocessing as mp
mp.set_start_method("fork", force=True)  # ensure fork, not spawn

import sys
import pathlib

# Add project root to PYTHONPATH so shared utils can be imported
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import datetime
import json
import pathlib
import time
import tqdm

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent_callback import LogObservationCallback
from utils.conf import Track_Infos, LOG_DIR

# Configuration
fixed_throttle = 0.24
track_index = 2  # jungle
logging     = False
steps       = 4000

if __name__ == '__main__':
    # Track and simulator settings
    track_info = Track_Infos[track_index]
    track      = track_info['track_name']
    sim_info   = track_info['simulator']
    ckpt_path  = track_info['model_path']  # .ckpt file

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Running for {steps} steps on track '{track}'")

    # Prepare logging directory
    daytime       = "day"
    weather       = "sunny"
    ts            = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    log_directory = pathlib.Path(LOG_DIR) / f"log_{ts}"
    log_directory.mkdir(parents=True, exist_ok=True)

    # Create simulator and gym
    assert pathlib.Path(sim_info['exe_path']).exists(), f"Simulator binary not found at {sim_info['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=sim_info['exe_path'],
        host=sim_info['host'],
        port=sim_info['port'],
    )
    env = UdacityGym(simulator=simulator)

    # Use Unity’s built-in SocketIO server
    simulator.start = simulator.sim_executor.start
    simulator.start()

    # Reset environment and wait until ready
    observation, _ = env.reset(track=track, weather=weather, daytime=daytime)
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    # Logging callback
    log_cb = LogObservationCallback(log_directory)

    # Load PyTorch Lightning model
    from models.dave2_legacy.model import Dave2
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Dave2.load_from_checkpoint(str(ckpt_path))
    model.eval().to(device)

    # Transforms to match training
    transform = T.Compose([
        T.Resize((160, 320)),  # ensure correct size
        T.AugMix(),            # if used in training
        T.ToTensor(),          # PIL -> [0,1]
    ])

    info = None
    try:
        for _ in tqdm.tqdm(range(steps)):
            img = observation.input_image  # PIL.Image
            x = transform(img).unsqueeze(0).to(device)  # (1,3,160,320)

            with torch.no_grad():
                out = model(x)
                steer = float(out.view(-1)[0].cpu())  # single output

            steer = np.clip(steer, -1.0, 1.0)
            action = UdacityAction(steering_angle=steer, throttle=fixed_throttle)

            last_obs = observation
            observation, reward, terminated, truncated, info = env.step(action)
            while observation.time == last_obs.time:
                observation = env.observe()
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("Interrupted by user. Saving logs and exiting...")

    finally:
        if info:
            with open(log_directory / "info.json", "w") as f:
                json.dump(info, f)
        if logging and log_cb.logs:
            log_cb.save()
            print(f"Logs saved to {log_directory}")
        else:
            print("No observations recorded — nothing to save.")
        simulator.close()
        env.close()
        print("Experiment concluded.")
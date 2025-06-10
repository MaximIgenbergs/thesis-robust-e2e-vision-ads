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
import torch
import torchvision.transforms as T
from pathlib import Path
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import UdacityAgent
from udacity_gym.agent_callback import LogObservationCallback
from utils.conf import Track_Infos
from models.utils.utils import make_collection_dir
from models.utils.device_config import DEFAULT_DEVICE

# Configuration
collector = 'vit'
track_index = 2  # jungle
logging = False
steps = 4000
model_path = PROJECT_DIR / 'models/vit/models/final_model.ckpt'
model_class_path = 'models.vit.model.ViT'  # Import path for the model class (string, not Path)

class TorchLightningAgent(UdacityAgent):
    """
    PyTorch Lightning agent for dual-output models (steering + throttle).
    """
    def __init__(self, model_path, model_class_path, device, 
                 before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        
        assert model_path.exists(), f"Model checkpoint not found: {model_path}"
        
        # Dynamically import the model class
        module_path, class_name = model_class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # Load the model from checkpoint
        self.model = model_class.load_from_checkpoint(str(model_path))
        self.model.eval().to(device)
        self.device = device
        
        # Standard transforms - adjust if your model needs different preprocessing
        self.transform = T.Compose([
            T.Resize((160, 320)),  # Resize to expected input size
            T.ToTensor(),          # Convert PIL to tensor [0,1]
        ])
        
        print(f"Loaded model: {model_class.__name__}")
        print(f"Device: {device}")

    def action(self, observation, *args, **kwargs) -> UdacityAction:
        img = observation.input_image  # PIL Image
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        with torch.no_grad():
            output = self.model(x)  # Expected shape: (1, 2) for [steering, throttle]
            
            steer = float(output[0, 0].cpu())
            throttle = float(output[0, 1].cpu())
        
        # Clamp values to valid ranges
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        
        return UdacityAction(steering_angle=steer, throttle=throttle)

if __name__ == '__main__':
    # Track & simulator settings
    track_info = Track_Infos[track_index]
    track = track_info['track_name']
    sim_info = track_info['simulator']
    daytime = 'day'
    weather = 'sunny'

    # Create a new collection directory under data/collections
    if logging:
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
    if logging:
        log_cb = LogObservationCallback(log_directory)

    # Instantiate agent
    agent = TorchLightningAgent(
        model_path=model_path,
        model_class_path=model_class_path,
        device=torch.device(DEFAULT_DEVICE),
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
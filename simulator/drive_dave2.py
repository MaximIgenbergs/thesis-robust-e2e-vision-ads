import json
import pathlib
import time
import tqdm
import datetime
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent_tf import SupervisedAgent_tf
from udacity_gym.agent_torch import DaveUdacityAgent
from udacity_gym.agent_callback import LogObservationCallback
from utils.conf import PROJECT_DIR, Track_Infos, LOG_DIR

track_index = 2

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567

    # Track settings
    track = Track_Infos[track_index]['track_name']
    daytime = "day"
    weather = "sunny"
    ts = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    log_directory = pathlib.Path(LOG_DIR) / f"log_{ts}"
    log_directory.mkdir(parents=True, exist_ok=True)
    print(Track_Infos[track_index]['simulator'])

    # Creating the simulator wrapper
    assert pathlib.Path(Track_Infos[track_index]['simulator']['exe_path']).exists(), f"Simulator binary not found at {Track_Infos[track_index]['simulator']['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=Track_Infos[track_index]['simulator']['exe_path'],
        host=Track_Infos[track_index]['simulator']['host'],
        port=Track_Infos[track_index]['simulator']['port'],
    )

    

    # Creating the gym environment
    env = UdacityGym(simulator=simulator)
    observation, _ = env.reset(track=track, weather=weather, daytime=daytime)

    simulator.start()

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    log_observation_callback = LogObservationCallback(log_directory)
    agent = SupervisedAgent_tf(
        model_path=Track_Infos[track_index]['model_path'],
        max_speed=25,
        min_speed=6,
        predict_throttle=True
    )

    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(4000)):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    if info:
        json.dump(info, open(log_directory.joinpath("info.json"), "w"))

    log_observation_callback.save()
    simulator.close()
    env.close()
    print("Experiment concluded.")

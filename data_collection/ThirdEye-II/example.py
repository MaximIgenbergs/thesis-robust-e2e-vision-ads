import json
import pathlib
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent, PIDUdacityAgent_Angle
from udacity_gym.agent_tf import SupervisedAgent_tf
from udacity_gym.agent_callback import LogObservationCallback, PauseSimulationCallback, ResumeSimulationCallback
from utils.conf import PROJECT_DIR, Track_Infos

track_index = 2 # jungle
logging = True
steps = 6000 # enough for 1 lap in jungle (~ 5200 )

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567

    # Track settings
    track = Track_Infos[track_index]['track_name']
    daytime = "day" # TODO: add day night cycle, maybe 'daynight'
    weather = "sunny" #TODO: add weather cycle? can be changed in the simulator at runtime
    log_directory = pathlib.Path(f"udacity_dataset_lake_dave/{track}_{weather}_{daytime}")
    print(Track_Infos[track_index]['simulator'])

    # Creating the simulator wrapper
    assert pathlib.Path(Track_Infos[track_index]['simulator']['exe_path']).exists(), f"Simulator binary not found at {Track_Infos[track_index]['simulator']['exe_path']}"
    simulator = UdacitySimulator(
        sim_exe_path=Track_Infos[track_index]['simulator']['exe_path'],
        host=Track_Infos[track_index]['simulator']['host'],
        port=Track_Infos[track_index]['simulator']['port'],
    )

    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
    )

    # Use Unitys build-in simulator 
    simulator.start = simulator.sim_executor.start # so calling simulator.start() will only launch the SocketIO server

    simulator.start()
    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    log_observation_callback = LogObservationCallback(log_directory) 

    if logging:
        agent = PIDUdacityAgent_Angle(
            track=track,
            before_action_callbacks=[],
            after_action_callbacks=[log_observation_callback],
        )
    else:
        agent = PIDUdacityAgent_Angle(
            track=track,
            before_action_callbacks=[],
        )
        print("log_observation_callback disabled.")

    # Take steps
    for _ in tqdm.tqdm(range(steps)):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.0025)

    if info:
        json.dump(info, open(log_directory.joinpath("info.json"), "w"))


    if log_observation_callback.logs:
        log_observation_callback.save()
    else:
        print("No observations were logged.")

    simulator.close()
    env.close()
    print("Experiment concluded.")

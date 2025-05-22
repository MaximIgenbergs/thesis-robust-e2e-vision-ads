import datetime
import json
import pathlib
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent, PIDUdacityAgent_Angle
from udacity_gym.agent_tf import SupervisedAgent_tf
from udacity_gym.agent_callback import LogObservationCallback, PauseSimulationCallback, ResumeSimulationCallback
from utils.conf import Track_Infos

# Configuration
track_index = 2 # jungle
logging = True
steps = 6000 # 1 lap in jungle ~ 5200

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567

    # Track settings
    track = Track_Infos[track_index]['track_name']
    daytime = "day"
    weather = "sunny"
    ts = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    log_directory = pathlib.Path(f"logs/log_{ts}")
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
        print("Logging has disabled by user.")

    # Take steps
    try:
        for _ in tqdm.tqdm(range(steps)):
            action = agent(observation)
            last_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)

            # Wait until a new frame arrives
            while observation.time == last_observation.time:
                observation = env.observe()
                time.sleep(0.0025)

    except KeyboardInterrupt:
        print("Execution interrupted by user. Saving logs and exiting...")

    finally:
        if info:
            info_path = log_directory.joinpath("info.json")
            with open(info_path, "w") as f:
                json.dump(info, f)

        if log_observation_callback.logs:
            log_observation_callback.save()
            print(f"Logs saved to {log_directory}")
        else:
            print("No observations were recorded --> nothing to save.")

        # Cleanly shut down simulator and environment
        simulator.close()
        env.close()
        print("Experiment concluded.")

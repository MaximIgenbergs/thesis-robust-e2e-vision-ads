import copy
import pathlib
import time
import socket
import json
import base64
from io import BytesIO
from PIL import Image

from .global_manager import get_simulator_state

from .action import UdacityAction
from .logger import CustomLogger
from .observation import UdacityObservation
from .unity_process import UnityProcess


# TODO: it should extend an abstract simulator
class UdacitySimulator:

    def __init__(
            self,
            sim_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
            host: str = "127.0.0.1",
            cmd_port: int = 55001,
            telemetry_port: int = 56001,
            event_port: int = 57001,
    ):
        # Simulator path
        self.simulator_exe_path = sim_exe_path
        self.sim_process = UnityProcess()

        # Network settings
        self.host = host
        self.cmd_port = cmd_port
        self.tel_port = telemetry_port
        self.event_port = event_port

        # Logging & shared state
        self.logger = CustomLogger(str(self.__class__))
        self.sim_state = get_simulator_state()

        # Buffer for partial telemetry lines
        self._tel_buffer = b""

        # Open raw-TCP sockets
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd_sock.connect((host, cmd_port))
        self.cmd_sock.settimeout(10.0)

        self.tel_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tel_sock.connect((host, telemetry_port))
        self.tel_sock.settimeout(10.0)

        self.event_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.event_sock.connect((host, event_port))
        self.event_sock.settimeout(10.0)

        # Verify binary location
        if not pathlib.Path(sim_exe_path).exists():
            self.logger.error(f"Executable binary to the simulator does not exists. "
                              f"Check if the path {self.simulator_exe_path} is correct.")

    def step(self, action: UdacityAction):
        self.sim_state['action'] = action
        return self.observe()

    def observe(self):
        return self.sim_state['observation']

    # TODO: add a sync parameter in pause method. if sync, the method waits for the pause response
    def pause(self):
        # TODO: change 'pause' with constant
        self.sim_state['paused'] = True
        # TODO: this loop is to make an async api synchronous
        # We wait the confirmation of the pause command
        while self.sim_state.get('sim_state', '') != 'paused':
            # TODO: modify the sleeping time with constant
            # print("waiting for pause...")
            time.sleep(0.1)
        # self.logger.info("exiting pause")

    def resume(self):
        self.sim_state['paused'] = False
        # TODO: this loop is to make an async api synchronous
        # We wait the confirmation of the resume command
        while self.sim_state.get('sim_state', '') != 'running':
            # TODO: modify the sleeping time with constant
            time.sleep(0.1)

    # # TODO: add other track properties
    # def set_track(self, track_name):
    #     self.sim_state['track'] = track_name

    def reset(self, new_track_name: str = 'lake', new_weather_name: str = 'sunny', new_daytime_name: str = 'day'):
        observation = UdacityObservation(
            input_image=None,
            semantic_segmentation=None,
            position=(0.0, 0.0, 0.0),
            steering_angle=0.0,
            throttle=0.0,
            speed=0.0,
            cte=0.0,
            lap=0,
            sector=0,
            next_cte=0.0,
            time=-1,
            angle_diff=0.0
        )
        action = UdacityAction(
            steering_angle=0.0,
            throttle=0.0,
        )
        self.sim_state['observation'] = observation
        self.sim_state['action'] = action
        # TODO: Change new track name to enum
        self.sim_state['track'] = {
            'track': new_track_name,
            'weather': new_weather_name,
            'daytime': new_daytime_name,
        }
        self.sim_state['events'] = []
        self.sim_state['episode_metrics'] = None

        return observation, {}

    def start(self):
        # Start Unity simulation subprocess
        self.logger.info("Starting Unity process for Udacity simulator...")
        self.sim_process.start(
            sim_path=self.simulator_exe_path, headless=False, port=self.cmd_port
        )

    def close(self):
        self.sim_process.close()
        for sock in (getattr(self, 'cmd_sock', None),
                     getattr(self, 'tel_sock', None),
                     getattr(self, 'event_sock', None)):
            if sock:
                try:
                    sock.close()
                except:
                    pass

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


class UdacitySimulator:

    def __init__(
            self,
            sim_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
            host: str = "127.0.0.1",
            cmd_port: int = 55001,
            telemetry_port: int = 56001,
            event_port: int = 57001,
            others_port: int = 58001
    ):
        # Simulator path
        self.simulator_exe_path = sim_exe_path
        self.sim_process = UnityProcess()

        # Network settings
        self.host = host
        self.cmd_port = cmd_port
        self.tel_port = telemetry_port
        self.event_port = event_port
        self.others_port = others_port

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

        self.others_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.others_sock.connect((host, others_port))
        self.others_sock.settimeout(10.0)

        # Verify binary location
        if not pathlib.Path(sim_exe_path).exists():
            self.logger.error(f"Executable binary to the simulator does not exists. "
                              f"Check if the path {self.simulator_exe_path} is correct.")

    def step(self, action: UdacityAction):
        # Send control command
        cmd = {
            "command": "send_control",
            "steering_angle": action.steering_angle,
            "throttle": action.throttle
        }
        self.cmd_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        self.sim_state['action'] = action

        # Read one full JSON line from telemetry, skipping empty or malformed lines
        while True:
            # If we already have a complete line in the buffer, process it
            if b"\n" in self._tel_buffer:
                line, sep, rest = self._tel_buffer.partition(b"\n")
                self._tel_buffer = rest
                text = line.strip()
                if not text:
                    continue # empty or whitespace-only line, skip it
                if text.startswith(b"{"):
                    data = json.loads(text.decode("utf-8"))
                    break
                else:
                    continue # non-JSON line, skip it

            # Otherwise, receive more bytes
            chunk = self.tel_sock.recv(4096)
            if not chunk:
                raise ConnectionError("Telemetry socket closed")
            self._tel_buffer += chunk

        # Map telemetry JSON into an observation
        try:
            img = Image.open(BytesIO(base64.b64decode(data["image"])))
        except Exception:
            img = None

        obs = UdacityObservation(
            input_image=img,
            semantic_segmentation=None,
            position=(
                float(data["pos_x"]),
                float(data["pos_y"]),
                float(data["pos_z"])
            ),
            steering_angle=float(data.get("steering_angle", 0.0)),
            throttle=float(data.get("throttle", 0.0)),
            speed=float(data["speed"]) * 3.6,
            cte=float(data["cte"]),
            next_cte=float(data["next_cte"]),
            lap=int(data["lap"]),
            sector=int(data["sector"]),
            time=int(time.time() * 1000),
            angle_diff=float(data.get("angular_difference", 0.0))
        )
        self.sim_state["observation"] = obs
        return obs

    def observe(self):
        return self.sim_state['observation']

    def pause(self):
        cmd = {"command": "pause_sim"}
        self.event_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        self.sim_state['paused'] = True

    def resume(self):
        cmd = {"command": "resume_sim"}
        self.event_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        self.sim_state['paused'] = False


    def reset(self, new_track_name: str = 'lake',
                    new_weather_name: str = 'sunny',
                    new_daytime_name: str = 'day'):
        # Send start_episode
        evt = {
            "command": "start_episode",
            "track_name": new_track_name,
            "weather_name": new_weather_name,
            "daytime_name": new_daytime_name
        }
        self.event_sock.sendall((json.dumps(evt) + "\n").encode("utf-8"))

        # Wait for {"event":"episode_started"} on the event socket
        buf_evt = b""
        while b"\n" not in buf_evt:
            buf_evt += self.event_sock.recv(4096)
        line_evt, _, _ = buf_evt.partition(b"\n")
        msg = json.loads(line_evt.decode("utf-8"))
        if msg.get("event") != "episode_started":
            raise RuntimeError(f"Unexpected event during reset: {msg}")

        # Scene is loaded --> grab the first telemetry frame
        buf_tel = b""
        while b"\n" not in buf_tel:
            buf_tel += self.tel_sock.recv(4096)
        line_tel, _, _ = buf_tel.partition(b"\n")
        data = json.loads(line_tel.decode("utf-8"))

        # Build and return a real observation
        try:
            img = Image.open(BytesIO(base64.b64decode(data["image"])))
        except Exception:
            img = None

        obs = UdacityObservation(
            input_image=img,
            semantic_segmentation=None,
            position=(
                float(data["pos_x"]),
                float(data["pos_y"]),
                float(data["pos_z"])
            ),
            steering_angle=0.0,
            throttle=0.0,
            speed=float(data["speed"]) * 3.6,
            cte=float(data["cte"]),
            next_cte=float(data["next_cte"]),
            lap=int(data["lap"]),
            sector=int(data["sector"]),
            time=int(time.time() * 1000),
            angle_diff=float(data["angular_difference"])
        )
        self.sim_state["observation"] = obs
        return obs, {}
    
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
                     getattr(self, 'event_sock', None),
                     getattr(self, 'others_sock', None)):
            if sock:
                try:
                    sock.close()
                except:
                    pass

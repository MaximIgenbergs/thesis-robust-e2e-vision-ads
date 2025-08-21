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
        sim_exe_path: str,
        host: str,
        cmd_port: int,
        telemetry_port: int,
        event_port: int,
        others_port: int,
    ):
        self.simulator_exe_path = sim_exe_path
        self.sim_process = UnityProcess()

        self.host = host
        self._ego_car_id = None
        self.cmd_port = cmd_port
        self.tel_port = telemetry_port
        self.event_port = event_port
        self.others_port = others_port

        # line buffers
        self._tel_buffer = b""
        self._others_buffer = b""

        self.logger = CustomLogger(str(self.__class__))
        self.sim_state = get_simulator_state()

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
        # short poll for spawn_response so we can implement our own deadline
        self.others_sock.settimeout(0.1)

        if not pathlib.Path(sim_exe_path).exists():
            self.logger.error(
                f"Executable binary to the simulator does not exist. Check if the path {self.simulator_exe_path} is correct."
            )

    def spawn_cars(self, cars):
        """Spawn a list of cars using the server's batch API."""
        msg = {"command": "spawn_cars", "cars": cars}
        self.others_sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    def spawn_random_cars(self, amount: int):
        """Ask the server to spawn a number of random NPC vehicles."""
        msg = {"command": "spawn_random_cars", "randomCarAmount": int(amount)}
        self.others_sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    def spawn_ego(self, *, circuit_name: str, spawn_index: int):
        """Spawn the ego via the batch API (one car)."""
        car = {
            "command": "spawn_cars",
            "cars": [{
                "name": "Ego",
                "prefab_name": "Objects/Car",
                "autonomous": True,
                "requestedCarId": 1,
                "speed": 25.0,
                "layer": "Road",
                "waypoints": [circuit_name],
                "spawn_point": float(spawn_index),
                "offset": [0.0, 0.0, 0.0],
                "scale_Vektor": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0],
                "humanBehavior": 0.0,
                "waitingPoints": []
            }]
        }
        self.others_sock.sendall((json.dumps(car) + "\n").encode("utf-8"))

    def _read_spawn_response(self, requested_id: int, timeout=5.0):
        """
        Read lines from others_sock until we see {"command":"spawn_response"}.
        Return assignedCarId or None on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.others_sock.recv(4096)
                if chunk:
                    self._others_buffer += chunk
                else:
                    time.sleep(0.01)
            except socket.timeout:
                # nothing this tick; keep looping until deadline
                pass

            while b"\n" in self._others_buffer:
                line, _, self._others_buffer = self._others_buffer.partition(b"\n")
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if msg.get("command") == "spawn_response":
                    assigned = msg.get("assignedCarId")
                    self.logger.info(
                        f"spawn_response: assignedCarId={assigned}, requested={msg.get('requestedCarId')}"
                    )
                    return assigned

            time.sleep(0.01)

        self.logger.error(f"No spawn_response within {timeout}s (requested_id={requested_id}).")
        return None

    def step(self, action: UdacityAction):
        ego_id = self.sim_state.get("ego_id")
        if ego_id is None:
            self.logger.error("ego_id is unknown; refusing to send control. Did reset() succeed?")
            ego_id = 1  # last-resort fallback to avoid crashing

        cmd = {
            "command": "send_control",
            "carControll": {
                "carId": int(ego_id),
                "Indicator": "Straight",  # Left / Right / Straight
                "steering_angle": float(action.steering_angle),
                "throttle": float(action.throttle),
            }
        }
        self.cmd_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        self.sim_state["action"] = action

        # read one telemetry JSON line
        while True:
            if b"\n" in self._tel_buffer:
                line, _, rest = self._tel_buffer.partition(b"\n")
                self._tel_buffer = rest
                text = line.strip()
                if not text:
                    continue
                if text.startswith(b"{"):
                    data = json.loads(text.decode("utf-8"))
                    break
                continue
            chunk = self.tel_sock.recv(4096)
            if not chunk:
                raise ConnectionError("Telemetry socket closed")
            self._tel_buffer += chunk

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
                float(data["pos_z"]),
            ),
            steering_angle=float(data.get("steering_angle", 0.0)),
            throttle=float(data.get("throttle", 0.0)),
            speed=float(data["speed"]) * 3.6,
            cte=float(data["cte"]),
            next_cte=float(data["next_cte"]),
            lap=int(data["lap"]),
            sector=int(data["sector"]),
            time=int(time.time() * 1000),
            angle_diff=float(data.get("angular_difference", 0.0)),
        )
        self.sim_state["observation"] = obs
        return obs

    def observe(self):
        return self.sim_state["observation"]

    def pause(self):
        cmd = {"command": "pause_sim"}
        self.event_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        self.sim_state["paused"] = True

    def resume(self):
        cmd = {"command": "resume_sim"}
        self.event_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        self.sim_state["paused"] = False


    def reset(
        self,
        new_track_name: str = "lake",
        new_weather_name: str = "sunny",
        new_daytime_name: str = "day",
        ego_circuit_name: str | None = None,
        ego_spawn_index: int = 0,
        npc_cars=None,
        random_npc_cars: int = 0,
    ):
        evt = {
            "command": "start_episode",
            "track_name": new_track_name,
            "weather_name": new_weather_name,
            "daytime_name": new_daytime_name,
        }
        self.event_sock.sendall((json.dumps(evt) + "\n").encode("utf-8"))

        # wait for {"event":"episode_started"}
        buf_evt = b""
        while b"\n" not in buf_evt:
            buf_evt += self.event_sock.recv(4096)
        line_evt, _, _ = buf_evt.partition(b"\n")
        msg = json.loads(line_evt.decode("utf-8"))
        if msg.get("event") != "episode_started":
            raise RuntimeError(f"Unexpected event during reset: {msg}")

        # spawn entities
        if ego_circuit_name:
            self.spawn_ego(circuit_name=ego_circuit_name, spawn_index=ego_spawn_index)
        if npc_cars:
            self.spawn_cars(npc_cars)
        if random_npc_cars:
            self.spawn_random_cars(random_npc_cars)

        # read back ego id
        requested_id = 1
        ego_id = self._read_spawn_response(requested_id=requested_id, timeout=5.0)
        if ego_id is None:
            self.logger.error("Missing spawn_response; using requested_id as ego_id for now.")
            ego_id = requested_id
        self.sim_state["ego_id"] = ego_id

        # first telemetry frame
        buf_tel = b""
        while b"\n" not in buf_tel:
            buf_tel += self.tel_sock.recv(4096)
        line_tel, _, _ = buf_tel.partition(b"\n")
        data = json.loads(line_tel.decode("utf-8"))

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
                float(data["pos_z"]),
            ),
            steering_angle=0.0,
            throttle=0.0,
            speed=float(data["speed"]) * 3.6,
            cte=float(data["cte"]),
            next_cte=float(data["next_cte"]),
            lap=int(data["lap"]),
            sector=int(data["sector"]),
            time=int(time.time() * 1000),
            angle_diff=float(data["angular_difference"]),
        )
        self.sim_state["observation"] = obs
        return obs, {}

    def start(self):
        self.logger.info("Starting Unity process for Udacity simulator...")
        self.sim_process.start(
            sim_path=self.simulator_exe_path, headless=False, port=self.cmd_port
        )

    def close(self):
        self.sim_process.close()
        for sock in (
            getattr(self, "cmd_sock", None),
            getattr(self, "tel_sock", None),
            getattr(self, "event_sock", None),
            getattr(self, "others_sock", None),
        ):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

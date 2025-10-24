import base64
import time
from io import BytesIO
from multiprocessing import Process
from threading import Thread

import PIL
import eventlet
eventlet.monkey_patch()
import numpy as np
from PIL import Image
from flask import Flask
# ---- Hard compat for modern Flask/Werkzeug + old flask_socketio ----------------
import sys, types, flask
from types import SimpleNamespace
from werkzeug.local import LocalStack

# 1) Ensure _request_ctx_stack exists AND has a non-None .top with a .session attr
if not hasattr(flask, "_request_ctx_stack"):
    flask._request_ctx_stack = LocalStack()
if flask._request_ctx_stack.top is None:
    # push a dummy top so flask_socketio can assign .session on it
    flask._request_ctx_stack.push(SimpleNamespace(session={}))

# 2) Provide werkzeug.serving.run_with_reloader for old imports
try:
    from werkzeug.serving import run_with_reloader  # noqa: F401
except Exception:
    from werkzeug import _reloader as _wzr
    shim = types.ModuleType("werkzeug.serving")
    def run_with_reloader(*args, **kwargs):
        return _wzr.run_with_reloader(*args, **kwargs)
    shim.run_with_reloader = run_with_reloader
    sys.modules["werkzeug.serving"] = shim
# ------------------------------------------------------------------------------

import flask_socketio as fso

# ---- after: import flask_socketio as fso ----
# Give flask_socketio a persistent, non-None request ctx stack with a .session
class _DummyTop:
    def __init__(self):
        self.session = {}

class _DummyStack:
    def __init__(self):
        self._top = _DummyTop()
    @property
    def top(self):
        # Always return the same object so assignment sticks if they reuse it
        return self._top
    def push(self, obj=None):  # no-op
        pass
    def pop(self):             # no-op
        pass

# Overwrite the module-level symbol that flask_socketio uses internally
fso._request_ctx_stack = _DummyStack()


from .action import UdacityAction
from .logger import CustomLogger
from .observation import UdacityObservation

class UdacityExecutor:
    # TODO: avoid cycles

    def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 4567,
    ):
        # Simulator network settings
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.sio = fso.SocketIO(
            self.app,
            async_mode='eventlet',
            cors_allowed_origins="*",
            transports=['websocket'],
            manage_session=False,
        )
        # Socket IO callbacks
        self.sio.on('connect')(self.on_connect)
        self.sio.on('car_telemetry')(self.on_telemetry)
        self.sio.on('episode_metrics')(self.on_episode_metrics)
        self.sio.on('episode_events')(self.on_episode_events)
        self.sio.on('episode_event')(self.on_episode_event)
        self.sio.on('sim_paused')(self.on_sim_paused)
        self.sio.on('sim_resumed')(self.on_sim_resumed)

        # Simulator logging
        self.logger = CustomLogger(str(self.__class__))
        # Simulator
        from .simulator import get_simulator_state
        self.sim_state = get_simulator_state()
        # Manage connection in separate process
        self.client_thread = Process(target=self._start_server)
        self.client_thread.daemon = True

    def on_telemetry(self, data):

        # self.logger.info(f"Received data from udacity client: {data}")
        # TODO: check data image, verify from sender that is not empty
        try:
            input_image = Image.open(BytesIO(base64.b64decode(data["image"])))
        except PIL.UnidentifiedImageError:
            print("Front facing camera image UnidentifiedImageError.")
            input_image = None

        """try:
            semantic_segmentation = Image.open(BytesIO(base64.b64decode(data["semantic_segmentation"])))
        except PIL.UnidentifiedImageError:
            print("Segmentation camera image UnidentifiedImageError.")
            semantic_segmentation = None"""
        semantic_segmentation = None
        # print(f"Received data from udacity client: {data}" )
        # print(f"Received data from udacity client: angle {data['angl']} angular difference: {data['angular_difference']}")
        # angle = float(data["angl"]) if float(data["angl"])<180 else float(data["angl"])-360

        observation = UdacityObservation(
            input_image=input_image,
            semantic_segmentation=semantic_segmentation,
            position=(float(data["pos_x"]), float(data["pos_y"]), float(data["pos_z"])),
            steering_angle=float(self.sim_state.get('action', None).steering_angle),
            throttle=float(self.sim_state.get('action', None).throttle),
            lap=int(data['lap']),
            sector=int(data['sector']),
            speed=float(data["speed"]) * 3.6,  # conversion m/s to km/h
            cte=float(data["cte"]),
            next_cte=float(data["next_cte"]),
            time=int(time.time() * 1000),
            angle_diff= float(data['angular_difference'])
        )
        self.sim_state['observation'] = observation

        # Sending control
        self.send_control()

        if self.sim_state.get('paused', False):
            self.send_pause()
        else:
            self.send_resume()
        track_info = self.sim_state.get('track', None)
        if track_info:
            track, weather, daytime = track_info['track'], track_info['weather'], track_info['daytime']
            self.send_track(track, weather, daytime)
            self.sim_state['track'] = None

    def on_connect(self):
        self.logger.info("Udacity client connected")
        track_info = self.sim_state.get('track', None)
        # TODO: do it in a better way
        while not track_info:
            time.sleep(1)
            track_info = self.sim_state.get('track', None)
        track, weather, daytime = track_info['track'], track_info['weather'], track_info['daytime']
        self.send_track(track, weather, daytime)
        self.sim_state['track'] = None

    def on_sim_paused(self, data):
        self.sim_state['sim_state'] = 'paused'

    def on_sim_resumed(self, data):
        # TODO: change 'running' with ENUM
        self.sim_state['sim_state'] = 'running'

    def on_episode_metrics(self, data):
        self.logger.info(f"episode metrics {data}")
        self.sim_state['episode_metrics'] = data

    def on_episode_events(self, data):
        self.logger.info(f"episode events {data}")
        self.sim_state['events'] += [data]

    def on_episode_event(self, data):
        self.logger.info(f"episode event {data}")
        self.sim_state['events'] += [data]

    def send_control(self) -> None:
        # self.logger.info(f"Sending control")
        action: UdacityAction = self.sim_state.get('action', None)
        if action:
            self.sio.emit(
                "action",
                data={
                    "steering_angle": action.steering_angle.__str__(),
                    "throttle": action.throttle.__str__(),
                },
                skip_sid=True,
            )
            eventlet.sleep(0)

    def send_pause(self):
        self.sio.emit("pause_sim", skip_sid=True)

    def send_resume(self):
        self.sio.emit("resume_sim", skip_sid=True)

    def send_track(self, track, weather, daytime):
        self.sio.emit("end_episode", skip_sid=True)
        self.sio.emit("start_episode", data={
            "track_name": track,
            "weather_name": weather,
            "daytime_name": daytime,
        }, skip_sid=True)

    def start(self):
        # Start Socket IO Server in separate thread
        self.client_thread.start()

    def _start_server(self):
        self.sio.run(self.app, host=self.host, port=self.port)

    def close(self):
        self.sio.stop()


if __name__ == '__main__':
    sim_executor = UdacityExecutor()
    sim_executor.start()

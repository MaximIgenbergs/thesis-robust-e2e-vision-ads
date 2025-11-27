from __future__ import annotations

"""
TCP agent with optional PerturbationDrive image perturbations.

Controlled via env vars:
    TCP_PD_FUNC      perturbation function name (string)
    TCP_PD_SEVERITY  severity level (int, >0 enables PD)
"""

import os
from typing import Dict, Tuple

import cv2
import numpy as np

from leaderboard.autoagents import autonomous_agent  # type: ignore
from team_code.tcp_agent import TCPAgent as BaseTCPAgent  # type: ignore

from perturbationdrive import ImagePerturbation


def get_entry_point() -> str: # Required by CARLA Leaderboard to find the agent class
    return "PerturbationAgent"


class PerturbationAgent(BaseTCPAgent):
    """
    Wraps TCPAgent and injects PerturbationDrive before the forward pass.
    """

    def setup(self, path_to_conf_file: str) -> None:
        super().setup(path_to_conf_file)
        self._configure_pd()


    def _configure_pd(self) -> None:
        func = os.environ.get("TCP_PD_FUNC", "").strip()
        severity_str = os.environ.get("TCP_PD_SEVERITY", "0").strip()

        try:
            severity = int(severity_str)
        except ValueError:
            severity = 0

        if not func or severity <= 0:
            self.pd_enabled = False
            self.pd_controller = None
            self.pd_func = ""
            self.pd_severity = 0
            print(f"[PerturbationAgent] PD disabled (func='{func}', severity={severity}).")
            return

        # TCP RGB sensor is 256x900 (see tcp_agent.py)
        self.pd_controller = ImagePerturbation(funcs=[func], image_size=(256, 900))
        self.pd_func = func
        self.pd_severity = severity
        self.pd_enabled = True

        print(f"[PerturbationAgent] PD enabled: func={func}, severity={severity}")

    # ------------------------------------------------------------------ #

    def _apply_pd(self, image_bgra: np.ndarray) -> np.ndarray:
        if not self.pd_enabled or self.pd_controller is None:
            return image_bgra

        # Drop alpha, work in RGB, then reattach
        bgr = image_bgra[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pert_rgb = self.pd_controller.perturbation(
            rgb, self.pd_func, int(self.pd_severity)
        )

        pert_bgr = cv2.cvtColor(pert_rgb, cv2.COLOR_RGB2BGR)
        out = image_bgra.copy()
        out[:, :, :3] = pert_bgr
        return out

    # ------------------------------------------------------------------ #

    def run_step(self, input_data: Dict[str, Tuple[int, np.ndarray]], timestamp: float):
        if self.pd_enabled and "rgb" in input_data:
            frame, image = input_data["rgb"]
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 3:
                image = self._apply_pd(image)
                modified = dict(input_data)
                modified["rgb"] = (frame, image)
                input_data = modified

        return super().run_step(input_data, timestamp)

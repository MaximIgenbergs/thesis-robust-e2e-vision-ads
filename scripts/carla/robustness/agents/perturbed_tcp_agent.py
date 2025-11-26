from pathlib import Path
import os
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np

# Make external/TCP importable as a top-level package
ROOT = Path(__file__).resolve().parents[3]
TCP_ROOT = ROOT / "external" / "TCP"
if str(TCP_ROOT) not in sys.path:
    sys.path.insert(0, str(TCP_ROOT))

from leaderboard.autoagents import autonomous_agent  # type: ignore
from team_code.tcp_agent import TCPAgent as BaseTCPAgent  # type: ignore

try:
    from perturbationdrive import ImagePerturbation
except ImportError:
    ImagePerturbation = None


def get_entry_point() -> str:
    return "PerturbedTCPAgent"


class PerturbedTCPAgent(BaseTCPAgent):
    """
    TCPAgent with optional image perturbations injected into the 'rgb' sensor
    before the original tick/run_step logic. Configured via environment
    variables so the TCP config remains only about checkpoints, etc.

      TCP_PD_FUNC      name of the perturbation function (string, required)
      TCP_PD_SEVERITY  integer severity (>=1 enables PD, 0 disables)
    """

    def setup(self, path_to_conf_file: str) -> None:
        super().setup(path_to_conf_file)
        self.build_pd_controller()

    def build_pd_controller(self) -> None:
        func = os.environ.get("TCP_PD_FUNC", "").strip()
        severity_str = os.environ.get("TCP_PD_SEVERITY", "0").strip()

        try:
            severity = int(severity_str)
        except ValueError:
            severity = 0

        if not func or severity <= 0 or ImagePerturbation is None:
            self.pd_enabled = False
            self.pd_func = None
            self.pd_severity = 0
            self.pd_controller = None
            if ImagePerturbation is None:
                print("[PerturbedTCPAgent] perturbationdrive not available, running clean.")
            else:
                print("[PerturbedTCPAgent] PD disabled (no TCP_PD_FUNC or severity <= 0).")
            return

        self.pd_func = func
        self.pd_severity = severity
        # TCP RGB sensor is height=256, width=900 (see sensors() in tcp_agent.py)
        self.pd_controller = ImagePerturbation(funcs=[func], image_size=(256, 900))
        self.pd_enabled = True
        print(f"[PerturbedTCPAgent] enabled: func={func} severity={severity}")

    def perturb_rgb_image(self, image_bgra: np.ndarray) -> np.ndarray:
        if not self.pd_enabled or self.pd_controller is None:
            return image_bgra

        bgr = image_bgra[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pert_rgb = self.pd_controller.perturbation(
            rgb, self.pd_func, int(self.pd_severity)
        )

        pert_bgr = cv2.cvtColor(pert_rgb, cv2.COLOR_RGB2BGR)
        out = image_bgra.copy()
        out[:, :, :3] = pert_bgr
        return out

    def run_step(
        self,
        input_data: Dict[str, Tuple[int, np.ndarray]],
        timestamp: float,
    ):
        if self.pd_enabled and "rgb" in input_data:
            frame, image = input_data["rgb"]
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 3:
                image_pert = self.perturb_rgb_image(image)
                modified = dict(input_data)
                modified["rgb"] = (frame, image_pert)
                input_data = modified

        return super().run_step(input_data, timestamp)

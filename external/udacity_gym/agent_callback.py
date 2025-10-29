import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pygame
import torch
import torchvision
import sys
from pathlib import Path

# add project root & perturbation-drive to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PD = ROOT / "external" / "perturbation-drive"
if str(PD) not in sys.path:
    sys.path.insert(0, str(PD))

try:
    # Reuse PD's preview window + overlays
    from perturbationdrive import ImageCallBack as _PDImageCallBack
except Exception:
    _PDImageCallBack = None


from external.udacity_gym import UdacityObservation, UdacitySimulator
from external.udacity_gym.logger import CustomLogger


class AgentCallback:

    def __init__(self, name: str, verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.logger = CustomLogger(str(self.__class__))

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        if self.verbose:
            self.logger.info(f"Activating callback {self.name}")


class PauseSimulationCallback(AgentCallback):

    def __init__(self, simulator: UdacitySimulator):
        super().__init__('stop_simulation')
        self.simulator = simulator

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        self.simulator.pause()


class ResumeSimulationCallback(AgentCallback):

    def __init__(self, simulator: UdacitySimulator):
        super().__init__('resume_simulation')
        self.simulator = simulator

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        self.simulator.resume()


class LogObservationCallback(AgentCallback):

    def __init__(self, path, enable_pygame_logging=False):
        super().__init__('log_observation')
        # Path initialization
        self.path = pathlib.Path(path)
        self.image_path = self.path.joinpath("image")
        self.segmentation_path = self.path.joinpath("segmentation")
        self.image_path.mkdir(parents=True, exist_ok=True)
        self.segmentation_path.mkdir(parents=True, exist_ok=True)
        self.logs = []
        self.logging_file = self.path.joinpath('log.csv')
        self.enable_pygame_logging = enable_pygame_logging
        if self.enable_pygame_logging:
            pygame.init()
            self.screen = pygame.display.set_mode((320, 160))
            camera_surface = pygame.surface.Surface((320, 160), 0, 24).convert()
            self.screen.blit(camera_surface, (0, 0))

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        metrics = observation.get_metrics()

        image_name = f"image_{observation.time:020d}.jpg"
        observation.input_image.save(self.image_path.joinpath(image_name))
        metrics['image_filename'] = image_name

        if observation.semantic_segmentation is not None:
            segmentation_name = f"segmentation_{observation.time:020d}.png"
            observation.semantic_segmentation.save(self.segmentation_path.joinpath(segmentation_name))
            metrics['segmentation_filename'] = segmentation_name

        if 'action' in kwargs.keys():
            metrics['predicted_steering_angle'] = kwargs['action'].steering_angle
            metrics['predicted_throttle'] = kwargs['action'].throttle
        if 'shadow_action' in kwargs.keys():
            metrics['shadow_predicted_steering_angle'] = kwargs['shadow_action'].steering_angle
            metrics['shadow_predicted_throttle'] = kwargs['shadow_action'].throttle
        self.logs.append(metrics)

        if self.enable_pygame_logging:
            pixel_array = np.swapaxes(np.array(observation.input_image), 0, 1)
            new_surface = pygame.pixelcopy.make_surface(pixel_array)
            self.screen.blit(new_surface, (0, 0))
            pygame.display.flip()

    def save(self):
        logging_dataframe = pd.DataFrame(self.logs)
        logging_dataframe = logging_dataframe.set_index('time', drop=True)
        logging_dataframe.to_csv(self.logging_file)
        if self.enable_pygame_logging:
            pygame.quit()


class TransformObservationCallback(AgentCallback):

    def __init__(self, transformation: Callable):
        super().__init__('transform_observation')
        self.transformation = transformation

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        super().__call__(observation, *args, **kwargs)
        augmented_image: torch.Tensor = self.transformation(observation.input_image,
                                                            mask=observation.semantic_segmentation, *args, **kwargs)
        image = torchvision.transforms.ToPILImage()(augmented_image.float())
        observation.input_image = image

        return observation


class PDPreviewCallback(AgentCallback):
    """
    Preview window callback that reuses PerturbationDrive's ImageCallBack.
    Call with: preview_cb(observation, display_image_np=<HxWx3 uint8>,
                          action=<UdacityAction>, perturbation=<str>)
    """
    def __init__(self, enabled: bool = True):
        super().__init__('pd_preview')
        self.enabled = enabled and (_PDImageCallBack is not None)
        self.monitor: Optional[_PDImageCallBack] = None # type: ignore
        if self.enabled:
            try:
                self.monitor = _PDImageCallBack()
                # optional splash if available
                if hasattr(self.monitor, "display_waiting_screen"):
                    self.monitor.display_waiting_screen()
            except Exception:
                self.monitor = None
                self.enabled = False

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        if not self.enabled or self.monitor is None:
            return

        img_np = kwargs.get("display_image_np", None)
        action = kwargs.get("action", None)
        pert   = kwargs.get("perturbation", "")

        if img_np is None:
            return

        steer_str = ""
        thr_str = ""
        if action is not None:
            try:
                steer_str = f"{float(action.steering_angle):.3f}"
                thr_str   = f"{float(action.throttle):.3f}"
            except Exception:
                pass

        # PD preview expects (image, steer_text, throttle_text, perturbation_name)
        try:
            self.monitor.display_img(img_np, steer_str, thr_str, pert)
        except Exception:
            # Non-fatal — keep driving
            pass

    def close(self):
        if self.monitor is not None:
            try:
                if hasattr(self.monitor, "display_disconnect_screen"):
                    self.monitor.display_disconnect_screen()
                self.monitor.destroy()
            except Exception:
                pass
            self.monitor = None
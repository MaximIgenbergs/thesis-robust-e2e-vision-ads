import pathlib
from typing import Callable

import numpy as np
import pandas as pd
import pygame
import torch
import torchvision

from udacity_gym import UdacityObservation, UdacitySimulator
from udacity_gym.logger import CustomLogger


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

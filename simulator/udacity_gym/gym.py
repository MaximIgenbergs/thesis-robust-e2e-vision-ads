from typing import Optional, Tuple, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action import UdacityAction
from .logger import CustomLogger
from .observation import UdacityObservation


class UdacityGym(gym.Env):
    """
    Gym interface for udacity simulator
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
            self,
            simulator,
            max_steering: float = 1.0,
            max_throttle: float = 1.0,
            input_shape: Tuple[int, int, int] = (3, 160, 320),
    ):
        # Save object properties and parameters
        self.simulator = simulator

        self.max_steering = max_steering
        self.max_throttle = max_throttle
        self.input_shape = input_shape

        self.logger = CustomLogger(str(self.__class__))

        # Initialize the gym environment
        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-max_steering, -max_throttle]),
            high=np.array([max_steering, max_throttle]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=input_shape, dtype=np.uint8
        )

    def step(
            self,
            action: UdacityAction
    ) -> tuple[UdacityObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle

        observation = self.simulator.step(action)

        # TODO: fix the two Falses
        return observation, observation.cte, False, False, {
            'events': self.simulator.sim_state['events'],
            'episode_metrics': self.simulator.sim_state['episode_metrics'],
        }

    def reset(self, **kwargs) -> tuple[UdacityObservation, dict[str, Any]]:

        # TODO: make reset synchronous
        # Returns only when the track has been set

        track = kwargs['track'] if 'track' in kwargs.keys() else 'lake'
        weather = kwargs['weather'] if 'weather' in kwargs.keys() else 'sunny'
        daytime = kwargs['daytime'] if 'daytime' in kwargs.keys() else 'day'
        observation, info = self.simulator.reset(track, weather, daytime)

        return observation, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.simulator.sim_state['observation'].image_array
        return None

    def observe(self) -> UdacityObservation:
        return self.simulator.observe()

    def close(self) -> None:
        if self.simulator is not None:
            self.simulator.close()

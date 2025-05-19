from typing import Union

from PIL import Image
import numpy as np


class UdacityObservation:

    def __init__(self,
                 input_image: Image.Image,
                 semantic_segmentation: Image.Image,
                 position: tuple[float, float, float],
                 steering_angle: float,
                 throttle: float,
                 speed: float,
                 cte: float,
                 next_cte: float,
                 lap: int,
                 sector: int,
                 time: int,
                 angle_diff: float
                 ):
        self.input_image = input_image
        self.semantic_segmentation = semantic_segmentation
        self.position = position
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.speed = speed
        self.cte = cte
        self.next_cte = next_cte
        self.lap = lap
        self.sector = sector
        self.time = time
        self.angle_diff = angle_diff

    def is_ready(self):
        # return self.input_image is not None and self.semantic_segmentation is not None
        return self.input_image is not None

    def get_metrics(self):
        return {
            'pos_x': self.position[0],
            'pos_y': self.position[1],
            'pos_z': self.position[2],
            'steering_angle': self.steering_angle,
            'speed': self.speed,
            'cte': self.cte,
            'lap': self.lap,
            'sector': self.sector,
            'next_cte': self.next_cte,
            'time': self.time,
            'angle_diff': self.angle_diff
        }

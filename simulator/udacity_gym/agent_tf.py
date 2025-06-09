import pathlib
import numpy as np
import os

from .agent import UdacityAgent
from .action import UdacityAction
from .observation import UdacityObservation

from tensorflow.keras.models import load_model # type: ignore
from utils.utils import preprocess
import tensorflow as tf
print("test is_gpu_available" ,tf.test.is_gpu_available())

class SupervisedAgent_tf(UdacityAgent):

    def __init__(
            self,
            model_path: str,
            max_speed: int,
            min_speed: int,
            predict_throttle: bool = False,
    ):
        super().__init__(before_action_callbacks=None, after_action_callbacks=None)

        assert os.path.exists(model_path), 'Model path {} not found'.format(model_path)

        self.model = load_model(model_path)
        self.predict_throttle = predict_throttle
        self.max_speed = max_speed
        self.min_speed = min_speed

    def action(self, observation: UdacityObservation, *args, **kwargs) -> UdacityAction:
        # observation by getting coordinate each time
        obs = observation.input_image # batch of images

        #print("Observations:", obs)
        obs = preprocess(obs)

        #  the model expects 4D array
        obs = np.array([obs])

        # obs = torch.transforms.Normalize(obs_mean,obs_std)
        speed = observation.speed

        if self.predict_throttle:
            action = self.model.predict(obs, batch_size=1, verbose=0)
            steering, throttle = action[0][0], action[0][1]
        else:
            import time
            time_start = time.time()
            steering = float(self.model.predict(obs, batch_size=1, verbose=0)[0])
            #print("DNN elasped time ",time.time() - time_start)
            steering = np.clip(steering, -1, 1)
            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed

            #steering = self.change_steering(steering=steering)
            #steering = float(self.model.predict(obs, batch_size=1, verbose=0))

            throttle = np.clip(a=1.0 - steering ** 2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)

            #print(f"steering {steering} throttle {throttle}")
            #self.model.summary()

        return UdacityAction(steering_angle=steering, throttle=throttle)
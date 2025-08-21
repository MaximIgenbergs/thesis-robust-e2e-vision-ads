import pathlib
import numpy as np
import os
from .action import UdacityAction
from .observation import UdacityObservation


class UdacityAgent:

    def __init__(self, before_action_callbacks=None, after_action_callbacks=None, transform_callbacks=None):
        self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
        self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []
        self.transform_callbacks = transform_callbacks if transform_callbacks is not None else []

    def on_before_action(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.before_action_callbacks:
            callback(observation, *args, **kwargs)

    def on_after_action(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.after_action_callbacks:
            callback(observation, *args, **kwargs)

    def on_transform_observation(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.transform_callbacks:
            observation = callback(observation, *args, **kwargs)
        return observation

    def action(self, observation: UdacityObservation, *args, **kwargs):
        raise NotImplementedError('UdacityAgent does not implement __call__')

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        if observation.input_image is None:
            return UdacityAction(steering_angle=0.0, throttle=0.0)
        self.on_before_action(observation)
        observation = self.on_transform_observation(observation)
        action = self.action(observation, *args, **kwargs)
        self.on_after_action(observation, action=action)
        return action


class PIDUdacityAgent(UdacityAgent):

    def __init__(self, kp, kd, ki, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.ki = ki  # Integral gain
        self.prev_error = 0.0
        self.total_error = 0.0

        self.curr_sector = 0
        self.skip_frame = 4
        self.curr_skip_frame = 0

    def action(self, observation: UdacityObservation, *args, **kwargs):

        if observation.sector != self.curr_sector:
            if self.curr_skip_frame < self.skip_frame:
                self.curr_skip_frame += 1
            else:
                self.curr_skip_frame = 0
                self.curr_sector = observation.sector
            error = observation.cte
        else:
            error = (observation.next_cte + observation.cte) / 2
        diff_err = error - self.prev_error

        # Calculate steering angle
        steering_angle = - (self.kp * error) - (self.kd * diff_err) - (self.ki * self.total_error)
        steering_angle = max(-1, min(steering_angle, 1))

        # Calculate throttle
        throttle = 1

        # Save error for next prediction
        self.total_error += error
        self.total_error = self.total_error * 0.99
        self.prev_error = error

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)


class PIDUdacityAgent_Angle(UdacityAgent):

    def __init__(self, #kp, kd, ki,
                 target_speed=30,
                 track="lake",  # different tracks have different control parameters
                 before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        # self.kp = kp  # Proportional gain
        # self.kd = kd  # Derivative gain
        # self.ki = ki  # Integral gain
        self.prev_error = 0.0
        self.total_error = 0.0
        self.prev_angle_error = 0.0
        self.total_angle_error = 0.0
        self.prev_speed_error = 0.0
        self.total_speed_error = 0.0

        self.curr_sector = 0
        self.skip_frame = 4
        self.curr_skip_frame = 0

        self.target_speed = target_speed

        self.track = track


    def action(self, observation: UdacityObservation, *args, **kwargs):
        # steer based on angle difference, throttle based on cte

        # if observation.sector != self.curr_sector:
        #     if self.curr_skip_frame < self.skip_frame:
        #         self.curr_skip_frame += 1
        #     else:
        #         self.curr_skip_frame = 0
        #         self.curr_sector = observation.sector
        #     cte = observation.cte
        # else:
        #     cte = (observation.next_cte + observation.cte) / 2

        cte = observation.cte
        angle_error = -observation.angle_diff

        diff_err = cte - self.prev_error

        speed_error=self.target_speed-observation.speed

        # Calculate steering angle
        # steering_angle = (-observation.angle_diff/180*math.pi)*0.5
        #steering_angle = - (self.kp * error) - (self.kd * diff_err) - (self.ki * self.total_error)

        if self.track == "lake":
            pid_parameters = {
                'Kp_angle': 0.007,
                'Kd_angle': 0.0014,
                'Ki_angle': 0.0,
                'Kp_speed': 0.1,
                'Kd_speed': 0.0,
                'Ki_speed': 0.0
            }
        elif self.track == "mountain":
            pid_parameters = {
                'Kp_angle': 0.005,
                'Kd_angle': 0.0014,
                'Ki_angle': 0.0,
                'Kp_speed': 0.1,
                'Kd_speed': 0.0,
                'Ki_speed': 0.0
            }
        elif self.track == "jungle":
            pid_parameters = {
                'Kp_angle': 0.01,
                'Kd_angle': 0.002,
                'Ki_angle': 0.0,
                'Kp_speed': 0.1,
                'Kd_speed': 0.0,
                'Ki_speed': 0.0
            }
        elif self.track == "city": # TODO: adjust for 90° turns
            pid_parameters = {
                'Kp_angle': 0.01,
                'Kd_angle': 0.002,
                'Ki_angle': 0.0,
                'Kp_speed': 0.1,
                'Kd_speed': 0.0,
                'Ki_speed': 0.0
            }
        elif self.track == "generator":
            pid_parameters = {
                'Kp_angle': 0.01,
                'Kd_angle': 0.002,
                'Ki_angle': 0.0,
                'Kp_speed': 0.1,
                'Kd_speed': 0.0,
                'Ki_speed': 0.0
            }

        (throttle, steering,
         #prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error
         ) \
             = pid_speed21(pid_parameters = pid_parameters,
                            road_error = observation.cte,
                           angle_error = angle_error,
                           speed_error = speed_error,
                           # prev_road_error = self.prev_error,
                           prev_angle_error = self.prev_angle_error,
                           prev_speed_error = self.prev_speed_error,
                           # total_road_error = self.total_error,
                           total_angle_error = self.total_angle_error,
                           total_speed_error = self.total_speed_error)

        steering_angle = steering
        print(f"steering_angle {steering_angle} Throttle {throttle}" )
        # Save error for next prediction
        self.prev_error = cte
        self.total_error += cte
        self.total_error = self.total_error * 0.99

        self.prev_angle_error = angle_error
        self.total_angle_error += angle_error

        self.prev_speed_error = speed_error
        self.total_speed_error += speed_error

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)


def pid_speed21(pid_parameters,
                road_error, angle_error, speed_error,
                prev_angle_error, prev_speed_error,
                total_angle_error, total_speed_error):

    road_error = -road_error
    if abs(angle_error) < 25 and abs(road_error) < 1:
        Kp_angle = pid_parameters['Kp_angle']
        Kd_angle = pid_parameters['Kd_angle']
    else:
        Kp_angle = pid_parameters['Kp_angle']*1.5
        Kd_angle = pid_parameters['Kd_angle']*1.5

    Ki_angle = pid_parameters['Ki_angle']

    Kp_speed = pid_parameters['Kp_speed']
    Ki_speed = pid_parameters['Ki_speed']
    Kd_speed = pid_parameters['Kd_speed']

    P_angle = Kp_angle * angle_error
    I_angle = Ki_angle * total_angle_error
    D_angle = Kd_angle * (angle_error - prev_angle_error)

    # P_road = Kp_road * road_error
    # I_road = Ki_road * total_road_error
    # D_road = Kd_road * (road_error - prev_road_error)

    steering = P_angle + I_angle + D_angle
    # steering = P_road + I_road + D_road + steering

    steering = np.clip(steering,-1,1) #  (-1, min(1, steering*0.4))

    P_speed = Kp_speed * speed_error
    I_speed = Ki_speed * total_speed_error
    D_speed = Kd_speed * (speed_error - prev_speed_error)
    throttle = P_speed + I_speed + D_speed
    throttle -= 0.1 * abs(road_error) + 0.05 * steering
    # throttle = max(-1, min(0.8, throttle))
    throttle = np.clip(throttle,-0.2,1)
    # print(f"s: {steering}, th: {throttle}, kp angle: {P_angle + I_angle + D_angle}, Kp road: {P_road + I_road + D_road}")

    return throttle, steering


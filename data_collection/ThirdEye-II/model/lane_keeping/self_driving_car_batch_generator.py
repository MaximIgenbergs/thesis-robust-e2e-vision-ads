import os

import numpy as np
from tensorflow.keras.utils import Sequence

from utils.conf import model_cfgs, Training_Configs
from utils.utils import load_image, augment, preprocess

class Generator(Sequence):

    def __init__(self, path_to_pictures, output, is_training, batch_size):
        self.path_to_pictures = path_to_pictures
        self.steering_angles = output[:,0]
        self.speed = output[:,1]
        self.is_training = is_training
        self.batch_size = batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        batch_paths = self.path_to_pictures[start_index:end_index]
        steering_angles = self.steering_angles[start_index:end_index]
        speed = self.speed[start_index:end_index]
        
        # print(f"Batch paths: {batch_paths}")
        # print(f"Steering angles: {steering_angles}")

        images = np.empty([len(batch_paths), model_cfgs['resized_image_height'], model_cfgs['resized_image_width'], model_cfgs['image_depth']])
        steers = np.empty([len(batch_paths)])
        
        # print(f"Initialized images array with shape: {images.shape}")
        # print(f"Initialized steers array with shape: {steers.shape}")

        for i, paths in enumerate(batch_paths):
            image = batch_paths[i]
            steering_angle = steering_angles[i]

            # print(f"Center: {center}, Left: {left}, Right: {right}, Steering angle: {steering_angle}")

            # augmentation
            if self.is_training and np.random.rand() < 0.6:
                image, steering = augment(Training_Configs['AUG'], image, steering_angle)
                # print(f"Augmented image shape: {image.shape}, Augmented steering angle: {steering_angle}")
            else:
                image = load_image(image)
                steering = steering_angle
                print(f"Loaded image shape: {image.shape}")

            images[i] = preprocess(image)
            # print(f"Preprocessed image shape: {images[i].shape}")
            steers[i] = steering
        output = np.stack((steers, speed), axis=1)

        return images, output

    def __len__(self):
        return len(self.path_to_pictures) // self.batch_size

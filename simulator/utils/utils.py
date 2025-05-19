import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# from utils.image_testing import print_image

from .conf import model_cfgs, PROJECT_DIR

RESIZED_IMAGE_HEIGHT= model_cfgs['resized_image_height']
RESIZED_IMAGE_WIDTH = model_cfgs['resized_image_width']
IMAGE_HEIGHT = model_cfgs['image_height']
IMAGE_WIDTH = model_cfgs['image_width']

def crop(image):
    return image[:-25, :, :]

def resize(image):
    """
    Resize the image to the input_image shape used by the network model
    """
    return cv2.resize(image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def normalize(image):
    """
    Normalize the image from all possible domains into [-1, 1]
    """
    max_val = image.max()
    min_val = image.min()

    if max_val >= 1.0 and max_val <= 255.0:
        image = image / 127.5 - 1.0
    elif min_val >= 0.0 and max_val <= 1.0:
        image = image * 2 - 1.0
    else:
        image = 2 * (image - min_val) / (max_val - min_val) - 1

    return image


def preprocess(image, debug = False):
    """
    Combine all preprocess functions into one
    """
    image = np.array(image)

    # image0 = (image * 255).astype('uint8')  # ! In this way, input image has to be [0,1]
    #print(f"max image 1 {image0.max()}")
    # pre-normalize to [0, 1]:
    # if max_val >= 1.0 and max_val <= 255.0:
    #     image = image/127.5 - 1.0
    # elif min_val >= 0 and max_val<= 1.0:
    #     image = image*2 - 1.0
    # else:
    #     image = 2 * (image-min_val)/(max_val - min_val) - 1
    image_crop = crop(image)
    image_resize = resize(image_crop)
    image_yuv = rgb2yuv(image_resize)

    image_nor = normalize(image_yuv)

    if debug:
        print(f"max image {image.max()} ")
        print(f"max image 2 {image_yuv.max()}")
        #print_image(image_0, image_resize, image_yuv, image_nor)
        plt.subplot(1,4,1)
        plt.imshow(image)
        plt.title(str(image.min()) +  "," + str(image.max()))
        plt.subplot(1,4,2)
        plt.imshow(image_crop)
        plt.title(str(image_crop.min())+  "," +str(image_crop.max()))
        plt.subplot(1,4,3)
        plt.imshow(image_resize)
        plt.title(str(image_resize.min())+  "," +str(image_resize.max()))
        plt.subplot(1,4,4)
        plt.imshow(image_nor)
        plt.title(str(round(image_nor.min(),2)) +  ","+ str(round(image_nor.max(),2)))
        plt.show()
    return image_nor

def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    #if np.random.rand() < 0.5 and steering_angle < 0:
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.4, high=0.7)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augment(training_configs_aug, image, steering_angle, range_x=50, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    #image, steering_angle = load_image(data_dir, center), steering_angle
    # if cfg.AUG_CHOOSE_IMAGE:
    #     image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    # else:
    image, steering_angle = load_image(image), steering_angle
    # TODO: flip should be applied to left/right only and w/ no probability
    if training_configs_aug['RANDOM_FLIP']:# and image in ["left", "right"]:
    #if cfg.AUG_RANDOM_FLIP:
        image, steering_angle = random_flip(image, steering_angle)
    if training_configs_aug['RANDOM_TRANSLATE']:
        image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    if training_configs_aug['RANDOM_SHADOW']:
        image = random_shadow(image)
    if training_configs_aug['RANDOM_BRIGHTNESS']:
        image = random_brightness(image)
    return image, steering_angle


def load_image(image_file):
    """
    Load RGB images from a file
    """

    target_str = "track"
    parts = image_file.split(target_str)
    if len(parts) > 1:
        result = PROJECT_DIR.joinpath(f"Data/lane_keeping_data/{target_str}{parts[1]}")
        img = np.asarray(Image.open(result)) # mpimg.imread(image_file)#(img_path)
        if img is not None:
            return img
    raise FileNotFoundError(image_file + " not found")

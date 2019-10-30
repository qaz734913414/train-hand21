import numpy as np
from easydict import EasyDict as edict

config = edict()
config.root = 'C:/train-hand21'

config.enable_blur = False
onfig.enable_black_border = False
config.enable_gray = False

config.EPS = 1e-14
config.min_rot_angle = -15
config.max_rot_angle = 15
config.landmark_img_set = 'hand143_panopticdb'

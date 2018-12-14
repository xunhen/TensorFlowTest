import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.CHANNEL = 16
__C.WINDOW = 300
__C.WINDOW_SHIFT = 50
__C.CLASS_NUM = 5
__C.TRAIN_DATA_RATIO = 0.6
__C.VAL_DATA_RATIO = 0.2
#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.TrainLogDir = 'TrainLog'
__C.TRAIN.DATA_PATH = 'DataSet/LS-0-1-3-8-9.txt'

__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.USE_EXP = True
__C.TRAIN.BASE_LR = 0.1
__C.TRAIN.LR_DECAY_RATE = 0.99
__C.TRAIN.LR_DECAY_STEPS = 1000
__C.TRAIN.LR_STAIRCASE = False

__C.TRAIN.USE_GradientDescent = True

__C.TRAIN.MAX_ITER = 20000
__C.TRAIN_DISPLAY_SUMMARY = 100
__C.TRAIN.EVAL_PER_STEP = 1000

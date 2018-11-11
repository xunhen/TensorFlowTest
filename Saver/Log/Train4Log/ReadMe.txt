from Tools.AttrDict import AttrDict
import os

__all__ = ['config']

config = AttrDict()
_C = config

_C.fake_data = False
_C.BaseDir = 'E:\CODE\Python\TensorFlowTest\Saver'
_C.DataSetDir = os.path.join(_C.BaseDir, 'DataSet')
_C.LogDir = os.path.join(_C.BaseDir, 'Log')
_C.TrainLogDir = os.path.join(_C.LogDir, 'Train4Log')
_C.TestLogDir = os.path.join(_C.LogDir, 'TestLog')
_C.FullShapes = [128, 32, 10]
_C.FullName = ['hidden1', 'hidden2', 'softmax']
_C.Conv = [32, 64, 128]
_C.Category = 10
_C.fc_num = 1024
_C.BatchSize = 128
_C.max_step = 15000
_C.stepsOfPeriod = 1000
_C.IMAGE_SIZE = 28
_C.IMAGE_PIXELS = _C.IMAGE_SIZE * _C.IMAGE_SIZE
_C.DECAY_BN = 0.99
_C.HAS_BN = True

_C.BaseLR = 0.00001  #0.0001 not useful
_C.LR_DECAY_RATE = 0.5
_C.LR_DECAY_STEPS = 1000

_C.LR_STAIRCASE=True

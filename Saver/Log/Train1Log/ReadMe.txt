lr 0.00001
from Tools.AttrDict import AttrDict
import os

__all__ = ['config']

config = AttrDict()
_C = config

_C.fake_data = False
_C.BaseDir = 'E:\CODE\Python\TensorFlowTest\Saver'
_C.DataSetDir = os.path.join(_C.BaseDir, 'DataSet')
_C.LogDir = os.path.join(_C.BaseDir, 'Log')
_C.TrainLogDir = os.path.join(_C.LogDir, 'Train1Log')
_C.TestLogDir = os.path.join(_C.LogDir, 'TestLog')
_C.FullShapes = [128, 32, 10]
_C.FullName = ['hidden1', 'hidden2', 'softmax']
_C.Conv=[32,64,128]
_C.Category = 10
_C.fc_num=512
_C.BatchSize = 128
_C.BaseLR = 0.00001
_C.max_step = 10000
_C.stepsOfPeriod = 1000
_C.IMAGE_SIZE = 28
_C.IMAGE_PIXELS = _C.IMAGE_SIZE * _C.IMAGE_SIZE


import os
import path
import numpy as np
from tensorflow.python import pywrap_tensorflow
def read(checkpoint_path):
    # read data from checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    data_print = np.array([])
    for key in var_to_shape_map:
        print('tensor_name', key)
        ckpt_data = np.array(reader.get_tensor(key))  # cast list to np arrary
        ckpt_data = ckpt_data.flatten()  # flatten list
        data_print = np.append(data_print, ckpt_data, axis=0)

    print(data_print, data_print.shape, np.max(data_print), np.min(data_print), np.mean(data_print))
    pass

if __name__ == '__main__':
    checkpoint_path = 'E:\\CODE\\Python\\TensorFlowTest\\Faster_RCNN\\vgg_16_2016_08_28\\vgg_16.ckpt'
    read(checkpoint_path)


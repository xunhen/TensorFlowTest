from MnistTest.Model import ModelBuilde_Conv
from MnistTest.Eval import placeholder_inputs, fill_feed_dict
from MnistTest.Config import config as cfg
import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np

def test():
    images_placeholder, labels_placeholder = placeholder_inputs(cfg.BatchSize, True)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(cfg.TrainLogDir + '/model.ckpt-5999.meta')
        saver.restore(sess, tf.train.latest_checkpoint(cfg.TrainLogDir))
        print(sess.run())


if __name__ == '__main__':
    x = [[0, 1, 0], [0, 0, 1]]
    ok = tf.nn.in_top_k(x, [1, 2], 1)
    max = tf.argmax(x, 1, output_type=dtypes.int32)
    ll = tf.equal([2, 2], max)
    img = tf.Variable(tf.random_normal([2,3, 4]))
    axis = list(range(len(img.get_shape()) - 1))
    mean, variance = tf.nn.moments(img, axis)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(img.eval())
        print(mean.eval(),variance.eval())


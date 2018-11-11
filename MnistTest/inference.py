from MnistTest.Model import ModelBuilde_Conv
from MnistTest.Eval import placeholder_inputs, fill_feed_dict
from MnistTest.Config import config as cfg
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def inference(input):
    logits = ModelBuilde_Conv(input)
    label = tf.nn.softmax(logits, name='softmax')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(cfg.TrainLogDir))
        y = sess.run([label])
        print(y)


if __name__ == '__main__':
    data_set = input_data.read_data_sets(cfg.DataSetDir)
    input_placeholder = tf.placeholder(tf.float32, shape=(1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 1), name='images')
    logits = ModelBuilde_Conv(input_placeholder, is_train=False)
    labels = tf.nn.softmax(logits, name='softmax')
    label = tf.argmax(labels, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(cfg.TrainLogDir))
        cfg.KEEP_PROB = 1.0
        for i in range(10):
            input = data_set.train.images[i].reshape([-1, 28, 28, 1])
            print(data_set.train.labels[i])
            y, ys = sess.run([label, labels], feed_dict={input_placeholder: input})
            print(ys)
            print(y)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from MnistTest.Config import config as cfg


def test_name_scope():
    with tf.name_scope('wjc') as scope:
        x = tf.constant(5.0, name='x')
        y = tf.constant(6.0, name='x')
        print('x.name', x.name)
        print('y.name', y.name)
        with tf.name_scope('wjc'):
            z = tf.constant(6.0, name='x')
            print('z.name', z.name)
            with tf.name_scope(''):
                h = tf.constant(6.0, name='x')
                print('h.name', h.name)
                with tf.name_scope(scope):
                    g = tf.constant(6.0, name='x')
                    print('g.name', g.name)
    with tf.name_scope('wjc'):
        k = tf.constant(6.0, name='x')
        print('k.name', k.name)
    with tf.name_scope('wjc'):
        b2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
        print('b2.name',b2.name)

def test_variable_scope():
    with tf.variable_scope('x'):
        w=tf.get_variable('weights',[1,3,3,5],initializer=tf.truncated_normal_initializer())
        print(w.name)
def test_variable_scope1():
    with tf.variable_scope('x',reuse=True):
        w=tf.get_variable('weights',[1,3,3,5],initializer=tf.truncated_normal_initializer())
        print(w.name)

def test_mix_scope():
    with tf.variable_scope('x'):
        x = tf.constant(5.0, name='x')
        y = tf.constant(6.0, name='x')
        print('x.name', x.name)
        print('y.name', y.name)
        w=tf.get_variable('weights',[1,3,3,5],initializer=tf.truncated_normal_initializer())
        print(w.name)
        with tf.variable_scope('x'):
            w = tf.get_variable('weights', [1, 3, 3, 5], initializer=tf.truncated_normal_initializer())
            print(w.name)
def test():
    test.index+=1
    print(test.index)
test.index=0

def str_test(s):
    s='213'
    return s

def testvar():
    with tf.variable_scope('pl'):
        images_placeholder = tf.placeholder(tf.float32, shape=(128, cfg.IMAGE_PIXELS), name='images')
        print(images_placeholder.name)
    with tf.variable_scope('pl'):
        images_placeholder = tf.placeholder(tf.float32, shape=(128, cfg.IMAGE_PIXELS), name='images')
        print(images_placeholder.name)
if __name__ == '__main__':
    #data_set = input_data.read_data_sets(cfg.DataSetDir)
    x=tf.nn.moments()

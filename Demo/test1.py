import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
slim.learning.train
slim.separable_conv2d
import numpy as np
PATH_TO_TEST_IMAGES_DIR = 'D:\Images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

import numpy as np
np.split

def test1():
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        plt.figure()
        plt.imshow(image)
    plt.show()

def test2():
    op = tf.Variable(tf.random_normal([2, 3, 4]))
    print(op.shape)
    print(op.get_shape())
    print(tf.shape(op))

    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])

    xx = tf.stack([x, y, z])
    yy = tf.stack([x, y, z], axis=1)
    with tf.Session() as sess:
        print(sess.run([xx, yy]))

def test3():
    x=tf.Variable([17.0],dtype=tf.float16,name='another_variable')
    saver=tf.train.Saver()
    init=tf.global_variables_initializer()
    save_path = 'F:\\DeepLearning\\222'
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess,save_path)

    anthor_Graph=tf.Graph()
    with anthor_Graph.as_default():
        x=tf.Variable([15.0], dtype=tf.float16, name='another_variable')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            uninitialized_vars_list = sess.run(tf.report_uninitialized_variables())
            print(uninitialized_vars_list)

            saver.restore(sess,save_path)
            print(sess.run(x))

            uninitialized_vars_list = sess.run(tf.report_uninitialized_variables())
            print(uninitialized_vars_list)
            pass

def test7():
    with tf.name_scope('test'):
        x=tf.placeholder(shape=[None,4],dtype=tf.int32)
        y=tf.ones_like(x)
        y=tf.reshape(y,shape=[-1,2])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        f=np.array([1,2,3,4,5,7,8,9])
        f=f.reshape([2,4])
        print(sess.run([x,y],feed_dict={x:f}))
def test8():
    with tf.name_scope('test'):
        x = tf.placeholder(shape=[None, 4], dtype=tf.int32)
        shape=tf.shape(x)
        s=x.get_shape().as_list()
        s=tf.constant(s)
        print(x.shape.as_list())
        print(not x.shape.as_list()[0])
        y=tf.Variable(tf.truncated_normal(shape=[2,1], stddev=1))
        print(y.shape.as_list())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        f = np.array([1, 2, 3, 4, 5, 7, 8, 9])
        f = f.reshape([2, 4])
        print(sess.run([x, shape,y,s], feed_dict={x: f}))
        print(x.shape.as_list())

if __name__ == '__main__':
    test8()

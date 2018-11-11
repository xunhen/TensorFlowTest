import tensorflow as tf


def variable_summary(name, var):
    with tf.name_scope('summary'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_sum(var)
        tf.summary.scalar(name + '/mean', mean)
        std = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar(name + '/std', std)

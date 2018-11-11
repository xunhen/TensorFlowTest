import tensorflow as tf
import math
from MnistTest.Summary import variable_summary
from MnistTest.Config import config as cfg

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


def generateFC(feature, input_shape, output_shape, name='fc', stddev=0.1, is_train=True, USE_BN=True, act=tf.nn.relu):
    if name == 'fc':
        if generateFC.index != 0:
            name = name + '_' + str(generateFC.index)
        generateFC.index += 1
    with tf.variable_scope(name):
        weights = tf.Variable(tf.truncated_normal([input_shape, output_shape], stddev=stddev), name='weights')
        biases = tf.get_variable('biased', shape=output_shape, initializer=tf.constant_initializer(0.1))
        feature = tf.nn.bias_add(tf.matmul(feature, weights), biases)
        variable_summary(name + '/after_conv_bias', feature)
        if USE_BN:
            if not cfg.HAS_DROPOUT and cfg.HAS_BN_AFTER_FC1:
                feature = tf.layers.batch_normalization(feature, training=is_train, name='bn_before_active')
                variable_summary('bn_after_fc', feature)
        hidden = act(feature)
        variable_summary(name + '/weights', weights)
        variable_summary(name + '/biases', biases)
        variable_summary(name + '/output', hidden)
        return hidden


generateFC.index = 0


def generateGroup(feature, conv_shape, name='group', stddev=0.1, act=tf.nn.relu, is_train=True):
    if name == 'group':
        if generateGroup.index != 0:
            name = name + '_' + str(generateGroup.index)
        generateGroup.index += 1
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=conv_shape,
                                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(feature, weights, [1, 1, 1, 1], padding='SAME', name='conv')
        if (cfg.HAS_BN):
            out_pre = tf.layers.batch_normalization(conv, training=is_train, name='bn_before_active')
            variable_summary(name + '/bn_before_active', out_pre)
        else:
            biases = tf.get_variable('biased', shape=conv_shape[3], initializer=tf.constant_initializer(0.1))
            out_pre = tf.nn.bias_add(conv, biases, name='output_before_active')
            variable_summary(name + '/biases', biases)
            variable_summary(name + '/output_before_active', out_pre)
        out = act(out_pre, name='output_after_active')
        variable_summary(name + '/weights', weights)
        variable_summary(name + '/conv', conv)
        variable_summary(name + '/output_after_active', out)
        return out


generateGroup.index = 0


def generateBN(feature, name='BN'):
    with tf.variable_scope(name):
        shape_param = feature.get_shape()[-1:]
        axis = range(len(feature.get_shape() - 1))
        beta = tf.get_variable('beta', shape_param, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', shape_param, initializer=tf.ones_initializer())
        moving_mean = tf.get_variable('moving_mean', shape_param, initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('moving_variance', shape_param, initializer=tf.ones_initializer(),
                                          trainable=False)

        mean, variance = tf.nn.moments(feature, axis, name='moment')
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, cfg.DECAY_BN)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, cfg.DECAY_BN)
        #####????


def generateMaxPool(feature, name='pool'):
    if name == 'pool':
        if generateMaxPool.index != 0:
            name = name + '_' + str(generateMaxPool.index)
        generateMaxPool.index += 1
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(feature, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        variable_summary(name + '/pool', pool)
        return pool


generateMaxPool.index = 0


def ModelBuild_Full(input, input_shape):
    feature = input
    last_shape = input_shape
    for shape, name in zip(cfg.FullShapes, cfg.FullName):
        feature = generateFC(feature, last_shape, shape, name)
        last_shape = shape
    return feature


def ModelBuilde_Conv(input, is_train=True):
    feature = input
    last_channel = 1
    for channel, conv_size in zip(cfg.Conv, cfg.Conv_Ksize):
        feature = generateGroup(feature, [conv_size, conv_size, last_channel, channel], is_train=is_train)
        if cfg.Conv_Double:
            feature = generateGroup(feature, [conv_size, conv_size, channel, channel], is_train=is_train)
        feature = generateMaxPool(feature)
        last_channel = channel
    shape = feature.get_shape().as_list()
    nodes = shape[1] * shape[2] * shape[3]
    feature = tf.reshape(feature, [shape[0], nodes])
    feature = generateFC(feature, nodes, cfg.fc_num, USE_BN=True, is_train=is_train)
    if cfg.HAS_DROPOUT:
        feature = tf.nn.dropout(feature, cfg.KEEP_PROB if is_train else 1.0)
        variable_summary('dropout_after_fc', feature)

    feature = generateFC(feature, cfg.fc_num, cfg.Category, USE_BN=False, is_train=is_train)
    return feature

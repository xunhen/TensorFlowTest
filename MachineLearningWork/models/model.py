import tensorflow as tf
from tensorflow.contrib import slim


class model(object):
    def __init__(self, class_num):
        self._class_num = class_num
        pass

    def extract_feature(self, input, is_training=True):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer,
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(input, 2, slim.conv1d, 32, 7, scope='conv1')
            net = slim.batch_norm(net, trainable=is_training, activation_fn=tf.nn.relu)
            net = slim.pool(net, 2, stride=2, pooling_type='MAX', scope='pool1')
            # net = slim.max_pool2d(net, (2, 2), stride=(2, 2), scope='pool1')

            net = slim.repeat(net, 3, slim.conv1d, 64, 7, scope='conv2')
            net = slim.batch_norm(net, trainable=is_training, activation_fn=tf.nn.relu)
            net = slim.pool(net, 2, stride=2, pooling_type='MAX', scope='pool2')
            # net = slim.max_pool2d(net, 2, stride=(2, 2), scope='pool1')

            net = slim.fully_connected(slim.flatten(net), 128, scope='fc3')
            net = slim.dropout(net, 0.5, scope='dropout3', is_training=is_training)
            net = slim.fully_connected(slim.flatten(net), 256, scope='fc4')
            net = slim.dropout(net, 0.5, scope='dropout4', is_training=is_training)

        return net

    def predict(self, features):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer,
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.fully_connected(slim.flatten(features), self._class_num, scope='fc5')
            net = slim.softmax(net)
        return net

    def loss(self, logits, labels):
        with tf.variable_scope('loss'):
            labels = tf.cast(labels, tf.int32)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            tf.summary.scalar('loss', loss)
        return loss

    def evaluation(self, logits, labels):
        with tf.variable_scope('eval'):
            correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
            return tf.reduce_sum(tf.cast(correct, tf.int32))

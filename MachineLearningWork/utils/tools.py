import tensorflow as tf
from MachineLearningWork.config import cfg
import numpy as np
from MachineLearningWork.models.model import model
from MachineLearningWork.DataSet.dataset import dataset


def fill_feed_dict(data_set, which, images_pl, labels_pl, reset=False, batch_size=cfg.TRAIN.BATCH_SIZE):
    images_feed, labels_feed = data_set.get_next_minibatch(batch_size=batch_size, style=which, reset=reset)
    images_feed = images_feed.tolist()
    for i in range(len(images_feed)):
        images_feed[i] = images_feed[i].tolist()
    images_feed = np.array(images_feed)
    images_feed = np.abs(images_feed)
    images_feed = images_feed.reshape([-1, cfg.WINDOW, cfg.CHANNEL])
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def placeholder_inputs(batchsize=cfg.TRAIN.BATCH_SIZE, isImage=True):
    with tf.variable_scope('placeholder'):
        if not isImage:
            images_placeholder = tf.placeholder(tf.float32, shape=(batchsize, cfg.WINDOW * cfg.CHANNEL),
                                                name='not_images')
        else:
            images_placeholder = tf.placeholder(tf.float32, shape=(batchsize, cfg.WINDOW, cfg.CHANNEL),
                                                name='images')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batchsize,), name='labels')
    return images_placeholder, labels_placeholder

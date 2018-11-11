import tensorflow as tf
from MnistTest.Config import config as cfg
def placeholder_inputs(batchsize=cfg.BatchSize,isImage=False):
    with tf.variable_scope('placeholder'):
        if not isImage:
            images_placeholder = tf.placeholder(tf.float32, shape=(batchsize, cfg.IMAGE_PIXELS), name='images')
        else:
            images_placeholder = tf.placeholder(tf.float32, shape=(batchsize, cfg.IMAGE_SIZE ,cfg.IMAGE_SIZE ,1),name='images')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batchsize,),name='labels')
    return images_placeholder, labels_placeholder
def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(cfg.BatchSize,cfg.fake_data)
    images_feed=images_feed.reshape([-1,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,1])
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples
    num_examples = steps_per_epoch * cfg.BatchSize
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
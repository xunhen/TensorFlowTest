import tensorflow as tf
from MachineLearningWork.utils.tools import fill_feed_dict
from MachineLearningWork.config import cfg


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            which,
            batch_size=cfg.TRAIN.BATCH_SIZE):
    true_count = 0

    steps_per_epoch = data_set.get_length(which) // batch_size
    num_examples = steps_per_epoch * batch_size
    reset = True
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, which, images_placeholder, labels_placeholder, reset)
        reset = False
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

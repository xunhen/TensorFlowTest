import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.examples.tutorials.mnist import mnist
from MnistTest.Model import ModelBuild_Full
from MnistTest.Model import ModelBuilde_Conv
from MnistTest.Eval import do_eval
from MnistTest.Summary import variable_summary
import time
import os

from MnistTest.Config import config as cfg


def placeholder_inputs(batchsize=cfg.BatchSize, isImage=False):
    with tf.variable_scope('placeholder'):
        if not isImage:
            images_placeholder = tf.placeholder(tf.float32, shape=(batchsize, cfg.IMAGE_PIXELS), name='images')
        else:
            images_placeholder = tf.placeholder(tf.float32, shape=(batchsize, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 1),
                                                name='images')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batchsize,), name='labels')
    return images_placeholder, labels_placeholder


def image_reshape(image):
    with tf.variable_scope('image_reshape'):
        input = tf.reshape(image, [-1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 1])
        tf.summary.image('image_reshape/input', input, max_outputs=10)


def loss(labels, logits):
    with tf.variable_scope('loss'):
        labels = tf.cast(labels, tf.int32)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar('loss', loss)
    return loss


def trainer(loss):
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if (cfg.USE_EXP):
            learning_rate = tf.train.exponential_decay(cfg.BaseLR, global_step, cfg.LR_DECAY_STEPS, cfg.LR_DECAY_RATE,
                                                       staircase=cfg.LR_STAIRCASE)
        else:
            learning_rate = cfg.BaseLR
        tf.summary.scalar('LR', learning_rate)
        if cfg.USE_GradientDescent:
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            opt = tf.train.AdamOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
    return train_op, global_step, learning_rate


def evaluation(logits, labels):
    with tf.variable_scope('eval'):
        correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
        return tf.reduce_sum(tf.cast(correct, tf.int32))


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(cfg.BatchSize, cfg.fake_data)
    images_feed = images_feed.reshape([-1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 1])
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def training(DataSets, isConv=False):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(cfg.BatchSize, True)
        if isConv:
            logits = ModelBuilde_Conv(images_placeholder)
        else:
            logits = ModelBuild_Full(images_placeholder, cfg.IMAGE_PIXELS)
        loss_op = loss(labels_placeholder, logits)
        train_op, global_step, lr = trainer(loss_op)

        eval = evaluation(logits, labels_placeholder)

        # only save trainable and bn variables
        var_list = tf.trainable_variables()
        if global_step is not None:
            var_list.append(global_step)
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        # save all variables
        # saver = tf.train.Saver(max_to_keep=5)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        total = 0

        with tf.Session() as sess:
            sess.run(init)
            train_summary = tf.summary.FileWriter(cfg.TrainLogDir, sess.graph)
            start_step = 0
            if tf.train.latest_checkpoint(cfg.TrainLogDir) is not None:
                print('restore begin...')
                saver.restore(sess, tf.train.latest_checkpoint(cfg.TrainLogDir))
                start_step = sess.run(global_step)
                print('restore end...')

            print('training begin!!')
            for step in range(start_step, cfg.max_step):
                start_time = time.time()
                feed_dict = fill_feed_dict(DataSets.train,
                                           images_placeholder,
                                           labels_placeholder)

                if (step + 1) % 100 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    if cfg.USE_EXP:
                        _, loss_value, summary_str, global_step_value, lr_value = sess.run(
                            [train_op, loss_op, summary, global_step, lr],
                            feed_dict=feed_dict, options=run_options,
                            run_metadata=run_metadata)
                    else:
                        _, loss_value, summary_str, global_step_value = sess.run(
                            [train_op, loss_op, summary, global_step],
                            feed_dict=feed_dict, options=run_options,
                            run_metadata=run_metadata)
                        lr_value = cfg.BaseLR
                    duration = time.time() - start_time
                    total += duration
                    print('Step %d: loss = %.5f (%.3f sec) lr = %.8f' % (step + 1, loss_value, duration, lr_value))
                    print('global_steps', global_step_value)
                    train_summary.add_run_metadata(run_metadata, 'step%03d' % global_step_value, global_step_value)
                    train_summary.add_summary(summary_str, global_step_value)
                    train_summary.flush()
                else:
                    _, loss_value, global_step_value = sess.run([train_op, loss_op, global_step], feed_dict=feed_dict)

                if (step + 1) % cfg.evalPerSteps == 0 or (step + 1) == cfg.max_step:
                    checkpoint_file = os.path.join(cfg.TrainLogDir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=global_step)
                    # print('Training Data Eval:')
                    # do_eval(sess, eval, images_placeholder, labels_placeholder, DataSets.train)
                    # print('Validation Data Eval:')
                    # do_eval(sess, eval, images_placeholder, labels_placeholder, DataSets.validation)
                    # print('Test Data Eval:')
                    # do_eval(sess, eval, images_placeholder, labels_placeholder, DataSets.test)
            if step + 1 == cfg.max_step:
                print('training times', total)
                print('training end!!')
            train_summary.close()


if __name__ == '__main__':
    data_set = input_data.read_data_sets(cfg.DataSetDir)
    training(data_set, True)

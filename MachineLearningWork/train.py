import tensorflow as tf
from MachineLearningWork.config import cfg
from MachineLearningWork.models.model import model
from MachineLearningWork.DataSet.dataset import dataset
from MachineLearningWork.utils.tools import fill_feed_dict, placeholder_inputs
from MachineLearningWork.eval import do_eval
import time
import os


def trainer(loss):
    with tf.variable_scope('train'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if cfg.TRAIN.USE_EXP:
            learning_rate = tf.train.exponential_decay(cfg.TRAIN.BASE_LR, global_step, cfg.TRAIN.LR_DECAY_STEPS,
                                                       cfg.TRAIN.LR_DECAY_RATE,
                                                       staircase=cfg.TRAIN.LR_STAIRCASE)
        else:
            learning_rate = cfg.BaseLR
        tf.summary.scalar('LR', learning_rate)
        if cfg.TRAIN.USE_GradientDescent:
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            opt = tf.train.AdamOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
    return train_op, global_step, learning_rate


def training(DataSets):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(cfg.TRAIN.BATCH_SIZE, True)
        model_build = model(cfg.CLASS_NUM)
        features = model_build.extract_feature(images_placeholder, True)
        logits = model_build.predict(features)
        loss_op = model_build.loss(logits, labels_placeholder)
        train_op, global_step, lr = trainer(loss_op)

        eval = model_build.evaluation(logits, labels_placeholder)

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
            train_summary = tf.summary.FileWriter(cfg.TRAIN.TrainLogDir, sess.graph)
            start_step = 0
            if tf.train.latest_checkpoint(cfg.TRAIN.TrainLogDir) is not None:
                print('restore begin...')
                saver.restore(sess, tf.train.latest_checkpoint(cfg.TRAIN.TrainLogDir))
                start_step = sess.run(global_step)
                print('restore end...')

            print('training begin!!')
            for step in range(start_step, cfg.TRAIN.MAX_ITER):
                start_time = time.time()
                feed_dict = fill_feed_dict(DataSets, 'train',
                                           images_placeholder,
                                           labels_placeholder)

                if (step + 1) % cfg.TRAIN_DISPLAY_SUMMARY == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    if cfg.TRAIN.USE_EXP:
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

                if (step + 1) % cfg.TRAIN.EVAL_PER_STEP == 0 or (step + 1) == cfg.TRAIN.MAX_ITER:
                    checkpoint_file = os.path.join(cfg.TRAIN.TrainLogDir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=global_step)
                    print('Training Data Eval:')
                    do_eval(sess, eval, images_placeholder, labels_placeholder, DataSets, 'train')
                    print('Validation Data Eval:')
                    do_eval(sess, eval, images_placeholder, labels_placeholder, DataSets, 'val')
                    print('Test Data Eval:')
                    do_eval(sess, eval, images_placeholder, labels_placeholder, DataSets, 'test')
            if step + 1 == cfg.TRAIN.MAX_ITER:
                print('training times', total)
                print('training end!!')
            train_summary.close()


if __name__ == '__main__':
    data_set = dataset(cfg.TRAIN.DATA_PATH, class_num=cfg.CLASS_NUM, channel=cfg.CHANNEL,
                       train_ratio=cfg.TRAIN_DATA_RATIO, val_ratio=cfg.VAL_DATA_RATIO
                       , window=cfg.WINDOW, windowshift=cfg.WINDOW_SHIFT)
    training(data_set)

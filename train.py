import json
import os
import time

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tqdm import tqdm

import data_utils as du
from config import MovieQAConfig
from get_dataset import MovieQAData
from model import VLLabMemoryModel

config = MovieQAConfig()


class TrainManager(object):
    def __init__(self, param):
        du.exist_make_dirs(config.exp_dir)
        self.param = param
        self.exp = {}
        self._load_exp()

    def train(self):
        if self.param.reset:
            self._new_exp()
            if os.path.exists(self._checkpoint_dir):
                os.system('rm -rf %s' % self._checkpoint_dir)
            if os.path.exists(self._log_dir):
                os.system('rm -rf %s' % self._log_dir)
            self._train()
        elif self.param.now_epoch < self.param.num_epochs - 1:
            self._train()
        else:
            print("The experiment of this setting finished.")

    def _train(self):
        du.exist_make_dirs(self._checkpoint_dir)
        du.exist_make_dirs(self._log_dir)
        start_time = time.time()
        now_epoch = self.param.now_epoch
        train_data = MovieQAData('train', modality=self.param.modality)
        eval_train_data = MovieQAData('train', modality=self.param.modality, is_training=False)
        val_data = MovieQAData('val', modality=self.param.modality, is_training=False)
        train_model, eval_train_model, val_model = self._get_model(train_data, eval_train_data, val_data)

        loss = tf.losses.sigmoid_cross_entropy(train_data.label,
                                               train_model.logits)

        val_loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(tf.one_hot(val_data.label,
                                                                             val_model.batch_size), 1),
                                                   val_model.logits)
        train_accu, train_accu_update, train_accu_init = \
            self._get_accuracy(train_model.prediction,
                               train_data.label,
                               'train_accuracy')

        eval_train_accu, eval_train_accu_update, eval_train_accu_init = \
            self._get_accuracy(tf.argmax(eval_train_model.prediction, 0),
                               eval_train_data.label,
                               'train_accuracy')

        val_accu, val_accu_update, val_accu_init = \
            self._get_accuracy(tf.argmax(val_model.prediction, 0),
                               val_data.label,
                               'val_accuracy')

        init_metric_op = tf.group(train_accu_init, val_accu_init, eval_train_accu_init)

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(config.initial_learning_rate,
                                                   global_step,
                                                   config.num_epochs_per_decay *
                                                   config.get_num_example(
                                                       'train',
                                                       self.param.modality,
                                                       is_training=True
                                                   ),
                                                   config.learning_rate_decay_factor,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        gradients, variables = list(zip(*grads_and_vars))
        # capped_grad_and_vars = [(g, v) if g is None else
        #                         (tf.clip_by_value(g, config.clip_gradients, -config.clip_gradients), v)
        #                         for g, v in grads_and_vars]
        gradients, _ = tf.clip_by_global_norm(gradients, config.clip_gradients)
        capped_grad_and_vars = list(zip(gradients, variables))
        train_op = optimizer.apply_gradients(capped_grad_and_vars, global_step)
        check_op = tf.add_check_numerics_ops()
        saver = tf.train.Saver(tf.global_variables(), )

        logging.info('Preparing training done with time %2f s', time.time() - start_time)

        # Summary
        train_summaries = []
        for var in tf.trainable_variables():
            train_summaries.append(tf.summary.histogram(var.name, var))
        train_summaries.append(tf.summary.scalar('train_loss', loss))
        train_summaries.append(tf.summary.scalar('train_accuracy', train_accu))
        train_summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        train_summaries_op = tf.summary.merge(train_summaries)

        eval_train_summaries = [tf.summary.scalar('eval_train_accuracy', eval_train_accu)]
        eval_train_summaries_op = tf.summary.merge(eval_train_summaries)

        val_summaries = [
            tf.summary.scalar('val_loss', val_loss),
            tf.summary.scalar('val_accu', val_accu)
        ]
        val_summaries_op = tf.summary.merge(val_summaries)

        checkpoint_file = tf.train.latest_checkpoint(self._checkpoint_dir)
        restore_fn = (lambda _sess: saver.restore(_sess, checkpoint_file)) \
            if checkpoint_file else None

        sv = tf.train.Supervisor(logdir=self._log_dir, summary_op=None,
                                 init_fn=restore_fn, save_model_secs=0,
                                 saver=saver, global_step=global_step)

        # clip_gradient_norm=self.config.clip_gradients)
        config_ = tf.ConfigProto(allow_soft_placement=True, )
        config_.gpu_options.allow_growth = True

        with sv.managed_session(config=config_) as sess:
            # Training loop
            def train_loop(epoch):
                logging.info("Training Loop Epoch %d", epoch + 1)
                try:
                    while True:
                        _, l, step, accu, pred \
                            = sess.run([train_op, loss, global_step, train_accu_update,
                                        train_model.prediction, ])
                        logging.info("[%s/%s] step: %d loss: %.3f accu: %.3f pred: %.2f, %.2f",
                                     epoch + 1, config.num_epochs, step, l, accu, pred[0], pred[1])
                        if step % 10 == 0:
                            summary = sess.run(train_summaries_op)
                            sv.summary_computed(sess, summary,
                                                tf.train.global_step(sess, global_step))
                        if step % 100 == 0:
                            sv.saver.save(sess, self._checkpoint_file,
                                          tf.train.global_step(sess, global_step))
                            eval_train_loop(epoch)
                            val_loop(epoch)

                except tf.errors.OutOfRangeError:
                    logging.info("Training Loop Epoch %d Done...", epoch + 1)
                    sv.saver.save(sess, self._checkpoint_file,
                                  tf.train.global_step(sess, global_step))
                except KeyboardInterrupt as e:
                    sv.saver.save(sess, self._checkpoint_file,
                                  tf.train.global_step(sess, global_step))
                    print()
                    raise e

            # Evaluation training loop
            def eval_train_loop(epoch):
                sess.run([eval_train_data.iterator.initializer, eval_train_accu_init], feed_dict={
                    eval_train_data.file_names_placeholder: eval_train_data.file_names,
                })
                accu = 0
                try:
                    with tqdm(total=config.get_num_example('train', self.param.modality)) as pbar:
                        while True:
                            accu = sess.run(eval_train_accu_update)
                            pbar.update()
                            pbar.set_description("[%s/%s] eval train accuracy: %.3f" %
                                                 (epoch + 1, config.num_epochs, accu))
                except tf.errors.OutOfRangeError:
                    summary = sess.run(eval_train_summaries_op)
                    sv.summary_computed(sess, summary,
                                        tf.train.global_step(sess, global_step))
                    logging.info("[%s/%s] evaluation train accuracy: %.3f", epoch + 1, config.num_epochs, accu)
                    logging.info("Evaluation Training Loop Epoch %d Done...", epoch + 1)
                except KeyboardInterrupt as e:
                    print()
                    raise e

            # Validation loop
            def val_loop(epoch):
                sess.run([val_data.iterator.initializer, val_accu_init], feed_dict={
                    val_data.file_names_placeholder: val_data.file_names,
                })
                accu = 0
                l = 0
                try:
                    with tqdm(total=config.get_num_example('val', self.param.modality)) as pbar:
                        while True:
                            l, accu = sess.run([val_loss, val_accu_update])
                            pbar.update()
                            pbar.set_description("[%s/%s] val accuracy: %.3f loss: %.3f" %
                                                 (epoch + 1, config.num_epochs, accu, l))
                except tf.errors.OutOfRangeError:
                    summary = sess.run(val_summaries_op)
                    sv.summary_computed(sess, summary,
                                        tf.train.global_step(sess, global_step))
                    logging.info("[%s/%s] validation accuracy: %.3f loss: %.3f", epoch + 1, config.num_epochs, accu, l)
                    logging.info("Evaluation Training Loop Epoch %d Done...", epoch + 1)

                except KeyboardInterrupt as e:
                    print()
                    raise e

            for epoch in range(now_epoch, config.num_epochs):
                self.param.now_epoch = epoch
                self._update_exp({
                    self._exp_name: self.param.__dict__
                })
                sess.run([train_data.iterator.initializer, init_metric_op,
                          eval_train_data.iterator.initializer,
                          val_data.iterator.initializer], feed_dict={
                    train_data.file_names_placeholder: train_data.file_names,
                    eval_train_data.file_names_placeholder: eval_train_data.file_names,
                    val_data.file_names_placeholder: val_data.file_names
                })
                train_loop(epoch)
                eval_train_loop(epoch)
                val_loop(epoch)

    def _get_model(self, train_data, eval_train_data, val_data, ):
        train_model = VLLabMemoryModel(train_data)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            val_model = VLLabMemoryModel(val_data, is_training=False)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            eval_train_model = VLLabMemoryModel(eval_train_data, is_training=False)
        return train_model, eval_train_model, val_model

    def _get_accuracy(self, predictions, label, name):
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, label, name=name)
        accuracy_init = tf.group(
            *[v.initializer for v in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name)])
        return accuracy, accuracy_update, accuracy_init

    @property
    def _log_dir(self):
        return os.path.join(config.log_dir, self._exp_name)

    @property
    def _checkpoint_dir(self):
        return os.path.join(config.checkpoint_dir, self._exp_name)

    @property
    def _checkpoint_file(self):
        return os.path.join(self._checkpoint_dir, self._exp_name)

    @property
    def _exp_name(self):
        name = [config.dataset_name, self.param.modality]
        for param in config.tunable_parameter.__dict__.keys():
            if self.param.__dict__.get(param, None) and \
                            config.tunable_parameter.__dict__[param] != self.param.__dict__[param]:
                name.append("%s_%s" % (param, self.param.__dict__[param]))
        return '_'.join(name)

    def _new_exp(self):
        self.param.now_epoch = 0
        self._update_now_exp()

    def _load_exp(self):
        if os.path.exists(config.exp_file):
            self.exp.update(json.load(open(config.exp_file, 'r')))
            if self.exp.get(self._exp_name, None):
                self.param.now_epoch = self.exp.get('now_epoch', 0)
            else:
                self._new_exp()
        else:
            self._new_exp()

    def _update_now_exp(self):
        self._update_exp({
            self._exp_name: self.param.__dict__
        })

    def _update_exp(self, item):
        self.exp.update(item)
        du.write_json(self.exp, config.exp_file)


def main(_):
    trainer = TrainManager(FLAGS)
    trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("modality", "fixed_num",
                        "fixed_num, fixed_interval, shot_major, subtitle_major")
    flags.DEFINE_integer("min_filter_size", 3, "")
    flags.DEFINE_integer("max_filter_size", 5, "")

    flags.DEFINE_integer("sliding_dim", 1024, "")
    # LSTM input and output dimensionality, respectively.
    flags.DEFINE_integer("embedding_size", 512, "")
    flags.DEFINE_integer("num_lstm_units", 512, "")

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    flags.DEFINE_float("lstm_dropout_keep_prob", 0.7, "")

    # Optimizer for training the model.
    flags.DEFINE_string("optimizer", "Adam", "")

    # Number of sliding convolution layer
    flags.DEFINE_integer("num_layers", 1, "")
    # Learning rate for the initial phase of training.
    flags.DEFINE_float("initial_learning_rate", 0.002, "")
    flags.DEFINE_float("learning_rate_decay_factor", 0.87, "")
    flags.DEFINE_float("num_epochs_per_decay", 1.0, "")

    # If not None, clip gradients to this value.
    flags.DEFINE_float("clip_gradients", 5.0, "")
    flags.DEFINE_bool("reset", False, "")
    # Number of epochs
    flags.DEFINE_integer("num_epochs", 20, "")
    FLAGS = flags.FLAGS
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

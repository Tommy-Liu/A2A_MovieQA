import hashlib
import json
import os
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import tf_logging as logging

from config import MovieQAConfig
from get_dataset import MovieQAData
from model import VLLabMemoryModel


class TrainManager(MovieQAConfig):
    def __init__(self):
        super(TrainManager, self).__init__()
        self.exp = {}
        self._load_exp()
        self._new_exp()

    def train(self):
        start_time = time.time()
        train_data = MovieQAData('train', dummy=True)
        val_data = MovieQAData('val', dummy=True)
        train_model, val_model = self._get_model(train_data, val_data)
        loss = tf.losses.sigmoid_cross_entropy(train_data.label,
                                               train_model.logits)

        val_loss = tf.losses.sigmoid_cross_entropy(val_data.label,
                                                   val_model.logits)
        train_accu, train_accu_update, train_accu_init = self._get_accuracy(train_model.prediction,
                                                                            train_data.label,
                                                                            'train_accuracy')
        val_accu, val_accu_update, val_accu_init = self._get_accuracy(val_model.prediction,
                                                                      val_data.label,
                                                                      'val_accuracy')
        init_metric_op = tf.group(train_accu_init, val_accu_init)

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                   global_step,
                                                   self.num_epochs_per_decay *
                                                   self.num_training_train_examples,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step, )
        saver = tf.train.Saver(tf.global_variables(), )

        logging.info('Preparing training done with time %2f s', start_time - time.time())

        # Summary
        train_summaries = []
        for var in tf.trainable_variables():
            train_summaries.append(tf.summary.histogram(var.name, var))
        train_summaries.append(tf.summary.scalar('train_loss', loss))
        train_summaries.append(tf.summary.scalar('train_accuracy', train_accu))
        train_summaries.append(tf.summary.scalar('learning_rate', learning_rate))
        train_summaries_op = tf.summary.merge(train_summaries)

        checkpoint_file = tf.train.latest_checkpoint(self._checkpoint_dir)

        def restore_fn(_sess):
            return saver.restore(_sess, checkpoint_file)

        sv = tf.train.Supervisor(logdir=self._log_dir, summary_op=None,
                                 init_fn=restore_fn, save_model_secs=0,
                                 saver=saver, global_step=global_step)

        # clip_gradient_norm=self.config.clip_gradients)
        config = tf.ConfigProto(allow_soft_placement=True, )
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            for i in range(self.num_epochs):
                sess.run([train_data.iterator.initializer, init_metric_op], feed_dict={
                    train_data.file_names_placeholder: train_data.file_names
                })
                logging.log("Training Loop Epoch %d", i)
                try:
                    while True:
                        _, l, step, accu = sess.run([train_op, loss, global_step, train_accu_update])
                        logging.info("[%s/%s] loss: %.3f accu: %.3f", i + 1, self.num_epochs, l, accu)
                        if step % 10 == 0:
                            summary = sess.run(train_summaries_op)
                            sv.summary_computed(sess, summary, global_step)
                        if step % 1000 == 0:
                            sv.saver.save(sess, self._checkpoint_dir, global_step)
                            val_accu_init.run()
                            while True:
                                l, accu = sess.run([val_loss, val_accu])

                except tf.errors.OutOfRangeError:
                    pass
                except KeyboardInterrupt:
                    pass
                finally:
                    sv.saver.save(sess, self._checkpoint_dir, global_step)

    def _get_model(self, train_data, val_data, ):
        train_model = VLLabMemoryModel(train_data)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            val_model = VLLabMemoryModel(val_data)
        return train_model, val_model

    def _get_accuracy(self, predictions, label, name):
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, label, name=name)
        accuracy_init = tf.group(
            *[v.initializer for v in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=name)])
        return accuracy, accuracy_update, accuracy_init

    @property
    def _log_dir(self):
        return os.path.join(self.log_dir, self._exp_hash_code)

    @property
    def _checkpoint_dir(self):
        return os.path.join(self.checkpoint_dir, self._exp_hash_code)

    @property
    def _exp_hash_code(self):
        param = json.dumps(self.tunable_parameter.__dict__, indent=4)
        h = hashlib.new('ripemd160')
        h.update(param.encode())
        return h.hexdigest()

    def _new_exp(self):
        self._update_exp({
            self._exp_hash_code: self.tunable_parameter.__dict__
        })

    def _load_exp(self):
        if os.path.exists(self.exp_file):
            self.exp.update(json.load(open(self.exp_file, 'r')))

    def _update_exp(self, item):
        self.exp.update(item)
        json.dump(self.exp, open(self.exp_file, 'w'), indent=4)


def main(_):
    trainer = TrainManager()
    trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("modality", "fixed_num",
                        "fixed_num, fixed_interval, shot_major, subtitle_major")
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

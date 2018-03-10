import json
import os
import time

import tensorflow as tf
from tqdm import trange

from config import MovieQAConfig
from get_dataset import MovieQAData
from model import VLLabMemoryModel
from utils import data_utils as du
from utils import func_utils as fu

config = MovieQAConfig()


class TrainManager(object):
    def __init__(self, param):
        fu.make_dirs(config.exp_dir)
        self.param = param
        self.exp = {}
        self._load_exp()

    def train(self):
        if self.param.reset:
            self._new_exp()
            if os.path.exists(self._checkpoint_dir):
                os.system('rm -rf %s' % os.path.join(self._checkpoint_dir, '*'))
            if os.path.exists(self._log_dir):
                os.system('rm -rf %s' % os.path.join(self._log_dir, '*'))
            self._train()
        elif self.param.now_epoch < self.param.num_epochs - 1:
            self._train()
        else:
            print("The experiment of this setting finished.")

    def _train(self):
        fu.make_dirs(self._checkpoint_dir)
        fu.make_dirs(self._log_dir)
        start_time = time.time()

        now_epoch = self.param.now_epoch
        num_train = config.get_num_example('train', self.param.modality, is_training=True)
        num_eval_train = config.get_num_example('train', self.param.modality)
        num_val = config.get_num_example('val', self.param.modality)

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
            self._get_accuracy(tf.round(train_model.prediction),
                               train_data.label,
                               'train_accuracy')

        eval_train_accu, eval_train_accu_update, eval_train_accu_init = \
            self._get_accuracy(tf.argmax(eval_train_model.prediction, 0),
                               eval_train_data.label,
                               'eval_train_accuracy')

        val_accu, val_accu_update, val_accu_init = \
            self._get_accuracy(tf.argmax(val_model.prediction, 0),
                               val_data.label,
                               'val_accuracy')

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(self.param.initial_learning_rate,
                                                   global_step,
                                                   self.param.num_epochs_per_decay * num_train,
                                                   self.param.learning_rate_decay_factor,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        gradients, variables = list(zip(*grads_and_vars))
        gradients, _ = tf.clip_by_global_norm(gradients, self.param.clip_gradients)
        capped_grad_and_vars = list(zip(gradients, variables))
        train_op = optimizer.apply_gradients(capped_grad_and_vars, global_step)
        saver = tf.train.Saver(tf.global_variables(), )

        print('Preparing training done with time %.2f s' % (time.time() - start_time))

        # Summary
        train_gv_summaries = []
        for idx, var in enumerate(variables):
            train_gv_summaries.append(tf.summary.histogram('gradient/' + var.name, gradients[idx]))
            train_gv_summaries.append(tf.summary.histogram(var.name, var))

        train_gv_summaries_op = tf.summary.merge(train_gv_summaries)

        train_summaries = [
            tf.summary.scalar('train_loss', loss),
            tf.summary.scalar('train_accuracy', train_accu),
            tf.summary.scalar('learning_rate', learning_rate)
        ]
        train_summaries_op = tf.summary.merge(train_summaries)

        eval_train_summaries = [tf.summary.scalar('eval_train_accuracy', eval_train_accu)]
        eval_train_summaries_op = tf.summary.merge(eval_train_summaries)

        val_summaries = [tf.summary.scalar('val_accu', val_accu)]
        val_summaries_op = tf.summary.merge(val_summaries)

        if self.param.checkpoint_file:
            checkpoint_file = self.param.checkpoint_file
        else:
            checkpoint_file = tf.train.latest_checkpoint(self._checkpoint_dir)
        restore_fn = (lambda _sess: saver.restore(_sess, checkpoint_file)) \
            if checkpoint_file else None

        sv = tf.train.Supervisor(logdir=self._log_dir, summary_op=None,
                                 init_fn=restore_fn, save_model_secs=0,
                                 saver=saver, global_step=global_step)

        config_ = tf.ConfigProto(allow_soft_placement=True, )
        # config_.gpu_options.allow_growth = True

        with sv.managed_session(config=config_) as sess:

            # Training loop
            def train_loop(epoch_):
                sess.run([train_data.iterator.initializer, train_accu_init], feed_dict={
                    train_data.file_names_placeholder: train_data.file_names,
                })
                step = tf.train.global_step(sess, global_step)
                print("Training Loop Epoch %d" % epoch_)
                if (step // num_train) < epoch_:
                    step = step % num_train

                for _ in range(step, config.get_num_example('train',
                                                            self.param.modality,
                                                            is_training=True)):
                    start_time = time.time()
                    try:
                        if step % 1000 == 0:
                            gv_summary, summary, _, l, step, accu, pred = sess.run(
                                [train_gv_summaries_op, train_summaries_op, train_op, loss, global_step,
                                 train_accu_update,
                                 train_model.prediction])
                            sv.summary_computed(sess, summary,
                                                tf.train.global_step(sess, global_step))
                            sv.summary_computed(sess, gv_summary,
                                                tf.train.global_step(sess, global_step))
                            sv.saver.save(sess, self._checkpoint_file,
                                          tf.train.global_step(sess, global_step))
                        elif step % 10 == 0:
                            summary, _, l, step, accu, pred = sess.run(
                                [train_summaries_op, train_op, loss, global_step, train_accu_update,
                                 train_model.prediction])
                            sv.summary_computed(sess, summary,
                                                tf.train.global_step(sess, global_step))
                        else:
                            _, l, step, accu, pred \
                                = sess.run([train_op, loss, global_step, train_accu_update,
                                            train_model.prediction])
                        print("[%s/%s] step: %d loss: %.3f accu: %.2f %% pred: %.2f, %.2f elapsed time: %.2f s" %
                              (epoch_, self.param.num_epochs, step, l, accu * 100,
                               pred[0], pred[1], time.time() - start_time))

                    except tf.errors.OutOfRangeError:
                        break
                    except KeyboardInterrupt:
                        sv.saver.save(sess, self._checkpoint_file,
                                      tf.train.global_step(sess, global_step))
                        print()
                        return True
                print("Training Loop Epoch %d Done..." % epoch_)
                sv.saver.save(sess, self._checkpoint_file,
                              tf.train.global_step(sess, global_step))
                return False

            # Evaluation training loop
            def eval_train_loop(epoch_):
                sess.run([eval_train_data.iterator.initializer, eval_train_accu_init], feed_dict={
                    eval_train_data.file_names_placeholder: eval_train_data.file_names,
                })
                accu = 0
                pbar = trange(num_eval_train)
                for _ in pbar:
                    try:
                        accu = sess.run(eval_train_accu_update)
                        pbar.set_description("[%s/%s] eval train accuracy: %.3f %%" %
                                             (epoch_, self.param.num_epochs, accu * 100))
                    except tf.errors.OutOfRangeError:
                        break
                    except KeyboardInterrupt:
                        print('')
                        pbar.close()
                        return True
                summary = sess.run(eval_train_summaries_op)
                sv.summary_computed(sess, summary,
                                    tf.train.global_step(sess, global_step))
                print("[%s/%s] evaluation train accuracy: %.3f %%" % (epoch_, self.param.num_epochs, accu * 100))
                print("Evaluation Training Loop Epoch %d Done..." % epoch_)
                pbar.close()
                return False

            # Validation loop
            def val_loop(epoch_):
                sess.run([val_data.iterator.initializer, val_accu_init], feed_dict={
                    val_data.file_names_placeholder: val_data.file_names,
                })
                accu = 0
                loss_ = 0
                pbar = trange(num_val)
                for _ in pbar:
                    try:
                        loss_, accu = sess.run([val_loss, val_accu_update])
                        pbar.set_description("[%s/%s] val accuracy: %.3f %% loss: %.3f" %
                                             (epoch_, self.param.num_epochs, accu * 100, loss_))
                    except tf.errors.OutOfRangeError:
                        break
                    except KeyboardInterrupt:
                        print('')
                        pbar.close()
                        return True
                summary = sess.run(val_summaries_op)
                sv.summary_computed(sess, summary,
                                    tf.train.global_step(sess, global_step))
                print("[%s/%s] validation accuracy: %.3f %% loss: %.3f" % (
                    epoch_, self.param.num_epochs, accu * 100, loss_))
                print("Validation Loop Epoch %d Done..." % epoch_)
                pbar.close()
                return False

            print(now_epoch, self.param.num_epochs)
            for epoch in range(now_epoch, self.param.num_epochs + 1):
                self.param.now_epoch = epoch
                self._update_now_exp()
                if train_loop(epoch) or \
                        eval_train_loop(epoch) or \
                        val_loop(epoch):
                    break

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
            if self.param.__dict__['__flags'].get(param, None) and \
                    config.tunable_parameter.__dict__[param] != self.param.__dict__['__flags'][param]:
                name.append("%s_%s" % (param, self.param.__dict__['__flags'][param]))
        return '-'.join(name)

    def _new_exp(self):
        self.param.now_epoch = 1
        self._update_now_exp()

    def _load_exp(self):
        if os.path.exists(config.exp_file):
            self.exp.update(json.load(open(config.exp_file, 'r')))
            if self.exp.get(self._exp_name, None):
                self.param.now_epoch = self.exp.get('now_epoch', 1)
            else:
                self._new_exp()
        else:
            self._new_exp()

    def _update_now_exp(self):
        self._update_exp({
            self._exp_name: self.param.__dict__['__flags']
        })

    def _update_exp(self, item):
        self.exp.update(item)
        du.json_dump(self.exp, config.exp_file)


def main(_):
    trainer = TrainManager(FLAGS)
    trainer.train()


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("modality", "fixed_num",
                        "There are 4 modalities of data sampling:" +
                        "fixed_num, fixed_interval, shot_major, subtitle_major." +
                        "Default: fixed_num")

    flags.DEFINE_integer("min_filter_size", 3,
                         "Minimal number of sliding window size." +
                         "Default: 3")

    flags.DEFINE_integer("max_filter_size", 5,
                         "Maximal number of sliding window size." +
                         "Default: 5")

    flags.DEFINE_integer("sliding_dim", 1024,
                         "The output dimension of sliding-window convolution." +
                         "Default: 1024")
    # LSTM input and output dimensionality, respectively.
    flags.DEFINE_integer("embedding_size", 512,
                         "The dimension of word embedding." +
                         "Default: 512")

    flags.DEFINE_integer("num_lstm_units", 512,
                         "The dimension of LSTM intenal state" +
                         "Default: 512")

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    flags.DEFINE_float("lstm_dropout_keep_prob", 0.7,
                       "The keeping probability of dropout layer." +
                       "Default: 0.7")

    # Optimizer for training the model.
    flags.DEFINE_string("optimizer", "Adam",
                        "The training optimization method." +
                        "Default: Adam")

    # Number of sliding convolution layer
    flags.DEFINE_integer("num_layers", 1,
                         "The number of the sliding-window convolution layer." +
                         "Default: 1")
    # Learning rate for the initial phase of training.
    flags.DEFINE_float("initial_learning_rate", 0.0001,
                       "The initial learning rate." +
                       "Default: 0.0001")

    flags.DEFINE_float("learning_rate_decay_factor", 0.87,
                       "The decay factor of learning rate." +
                       "Default: 0.87 (OPOP)")

    flags.DEFINE_float("num_epochs_per_decay", 1.0,
                       "The number of epochs when learning rate decays at." +
                       "Default: 1.0")

    # If not None, clip gradients to this value.
    flags.DEFINE_float("clip_gradients", 1.0,
                       "The value of global norm of gradient." +
                       "Default: 1.0")

    flags.DEFINE_bool("reset", False,
                      "Reset the experiment." +
                      "Default: False")
    # Number of epochs
    flags.DEFINE_integer("num_epochs", 20,
                         "The number of training epochs." +
                         "Default: 20")
    flags.DEFINE_string("checkpoint_file", "",
                        "The checkpoint you want to start from.")
    FLAGS = flags.FLAGS
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

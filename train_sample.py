import argparse
import importlib
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import trange

from config import MovieQAPath
# from input import Input as In
# from input_v2 import Input as In2
from soft_input import Input
from utils import data_utils as du
from utils import func_utils as fu
from utils import model_utils as mu

_mp = MovieQAPath()


class TrainManager(object):
    def __init__(self):
        if reset:
            if os.path.exists(self._checkpoint_dir):
                os.system('rm -rf %s' % self._checkpoint_dir)
            if os.path.exists(self._log_dir):
                os.system('rm -rf %s' % self._log_dir)
            if os.path.exists(self._attn_dir):
                os.system('rm -rf %s' % self._attn_dir)

        fu.make_dirs(os.path.join(self._checkpoint_dir, 'best'))
        fu.make_dirs(self._log_dir)
        fu.make_dirs(self._attn_dir)

        self.train_data = Input(split='train', mode=args.mode, drop_rate=drop_rate)
        self.train_data.drop()
        self.val_data = Input(split='val', mode=args.mode)
        # self.test_data = TestInput()

        self.train_model = mod.Model(self.train_data, beta=hp['reg'], training=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.val_model = mod.Model(self.val_data)

        # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        #     self.test_model = mod.Model(self.test_data, training=False)

        for v in tf.trainable_variables():
            print(v)

        self.main_loss = mu.get_loss(hp['loss'], self.train_data.gt, self.train_model.output)
        self.regu_loss = tf.losses.get_regularization_loss()

        self.loss = 0
        self.loss += self.main_loss
        self.loss += self.regu_loss

        self.train_answer = tf.argmax(self.train_model.output, axis=1)
        self.train_accuracy, self.train_accuracy_update, self.train_accuracy_initializer \
            = mu.get_acc(self.train_data.gt, self.train_answer, name='train_accuracy')

        self.val_answer = tf.argmax(self.val_model.output, axis=1)
        self.val_accuracy, self.val_accuracy_update, self.val_accuracy_initializer \
            = mu.get_acc(self.val_data.gt, tf.argmax(self.val_model.output, axis=1), name='val_accuracy')

        self.global_step = tf.train.get_or_create_global_step()

        decay_step = int(hp['decay_epoch'] * len(self.train_data))
        self.learning_rate = mu.get_lr(hp['decay_type'], hp['learning_rate'], self.global_step,
                                       decay_step, hp['decay_rate'])

        self.optimizer = mu.get_opt(hp['opt'], self.learning_rate, decay_step)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = tf.group(self.optimizer.apply_gradients(grads_and_vars, self.global_step),
                                     self.train_accuracy_update)

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables())

        self.train_init_op_list = [self.train_data.initializer, self.train_accuracy_initializer]

        self.val_init_op_list = [self.val_data.initializer, self.val_accuracy_initializer]

        self.train_op_list = [self.train_op, self.loss, self.train_accuracy, self.global_step]

        self.val_op_list = [self.val_accuracy, self.val_accuracy_update]

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=config) as sess:
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            try:
                for _ in range(repeat):
                    self.train_data.drop()
                    for _ in range(3):
                        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                        acc, train_max, val_max = 0, 0, 0

                        for epoch in range(32):
                            epoch = epoch + 1
                            # Train Loop
                            sess.run(self.train_init_op_list, feed_dict=self.train_data.feed_dict)

                            with trange(len(self.train_data)) as pbar:
                                for _ in pbar:
                                    _, l, acc, step = sess.run(self.train_op_list)
                                    pbar.set_description('[%03d] Train loss: %.3f acc: %.2f' % (epoch, l, acc))
                                if train_max < acc:
                                    train_max = acc
                            # Validation Loop
                            sess.run(self.val_init_op_list, feed_dict=self.val_data.feed_dict)
                            with trange(len(self.val_data)) as pbar:
                                for _ in pbar:
                                    acc, _ = sess.run(self.val_op_list)
                                    pbar.set_description('[%03d] Validation acc: %.2f' % (epoch, acc))
                                if acc > val_max:
                                    val_max = acc
                        self.record(self.train_data.index, train_max, val_max)
            except KeyboardInterrupt:
                print()

    def record(self, index, val_max, train_max):
        val_max = float(val_max)
        train_max = float(train_max)
        acc_file = '%s.json' % self._model_name
        h = str(integer_hash(index))
        if os.path.exists(acc_file):
            acc = du.json_load(acc_file)
            if acc.get(h, None) is None:
                acc[h] = [[val_max, train_max]]
            else:
                acc[h].append([val_max, train_max])
        else:
            acc = {h: [[val_max, train_max]]}
        du.json_dump(acc, acc_file)

    @property
    def _model_name(self):
        return '-'.join([args.mod, args.hp, args.extra])

    @property
    def _log_dir(self):
        return os.path.join(_mp.log_dir, self._model_name)

    @property
    def _checkpoint_dir(self):
        return os.path.join(_mp.checkpoint_dir, self._model_name)

    @property
    def _checkpoint_file(self):
        return os.path.join(self._checkpoint_dir, self._model_name)

    @property
    def _best_checkpoint(self):
        return os.path.join(self._checkpoint_dir, 'best', self._model_name)

    @property
    def _attn_dir(self):
        return os.path.join(_mp.attn_dir, self._model_name)


def integer_hash(index):
    h = 0
    for i in index:
        h += 1 << i
    return h


def integer_unhash(h):
    return [idx for idx, c in enumerate(bin(h)[2:]) if c == '1']


def main():
    trainer = TrainManager()
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod', default='model', help='Model used to train.')
    parser.add_argument('--reset', action='store_true', help='Reset the experiment.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--mode', default='feat+subt', help='Data mode we use.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint file.')
    parser.add_argument('--hp', default='01', help='Hyper-parameters.')
    parser.add_argument('--extra', default='', help='Extra model name.')
    parser.add_argument('--attn', action='store_true', help='Save attention.')
    parser.add_argument('--repeat', default=64, type=int, help='Regularize the model.')
    parser.add_argument('--drop_rate', default=0.4, type=float, help='Drop rate of data.')
    args = parser.parse_args()
    mod = importlib.import_module('model.' + args.mod)
    hp = getattr(importlib.import_module('hp'), 'hp' + args.hp)
    reset = args.reset
    debug = args.debug
    attn = args.attn
    repeat = args.repeat
    drop_rate = args.drop_rate
    main()

import argparse
import importlib
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import trange

from config import MovieQAPath
# from input import Input as In
# from input_v2 import Input as In2
from raw_input import Input
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

        self.train_data = Input(split='train', mode=args.mode)
        self.val_data = Input(split='val', mode=args.mode)
        # self.test_data = TestInput()

        self.train_model = mod.Model(self.train_data, beta=hp['reg'], training=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.val_model = mod.Model(self.val_data)

        # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        #     self.test_model = mod.Model(self.test_data, training=False)

        for v in tf.trainable_variables():
            print(v)

        self.train_attn = tf.squeeze(self.train_model.attn[self.train_data.gt[0]])
        self.val_attn = tf.squeeze(self.val_model.attn[self.val_data.gt[0]])
        self.main_loss = mu.get_loss(hp['loss'], self.train_data.gt, self.train_model.output)
        self.attn_loss = mu.get_loss('hinge', tf.to_float(self.train_data.spec), self.train_attn)
        self.regu_loss = tf.losses.get_regularization_loss()

        self.loss = 0
        if 'main' in target:
            self.loss += self.main_loss
        elif 'attn' in target:
            self.loss += self.attn_loss
        self.loss += self.regu_loss

        self.train_acc, self.train_acc_update, self.train_acc_init = \
            mu.get_acc(self.train_data.gt, tf.argmax(self.train_model.output, axis=1), name='train_accuracy')

        self.train_attn_acc, self.train_attn_acc_update, self.train_attn_acc_init = \
            mu.get_acc(self.train_data.spec, tf.to_int32(self.train_attn > 0.5), name='train_attention_accuracy')

        # self.train_q_attn_acc, self.train_q_attn_acc_update, self.train_q_attn_acc_init = \
        #     tf.metrics.accuracy(self.train_data.spec, self.train_model.output, name='train_q_attention_accuracy')
        #
        # self.train_a_attn_acc, self.train_a_attn_acc_update, self.train_a_attn_acc_init = \
        #     tf.metrics.accuracy(self.train_data.spec, self.train_model.output, name='train_a_attention_accuracy')

        self.val_acc, self.val_acc_update, self.val_acc_init = \
            mu.get_acc(self.val_data.gt, tf.argmax(self.val_model.output, axis=1), name='val_accuracy')

        self.val_attn_acc, self.val_attn_acc_update, self.val_attn_acc_init = \
            mu.get_acc(self.val_data.spec, tf.to_int32(self.val_attn > 0.5), name='val_attention_accuracy')

        # self.val_q_attn_acc, self.val_q_attn_acc_update, self.val_q_attn_acc_init = \
        #     tf.metrics.accuracy(self.train_data.spec, self.train_model.output, name='val_q_attention_accuracy')
        #
        # self.val_a_attn_acc, self.val_a_attn_acc_update, self.val_a_attn_acc_init = \
        #     tf.metrics.accuracy(self.train_data.spec, self.train_model.output, name='val_a_attention_accuracy')

        self.global_step = tf.train.get_or_create_global_step()

        decay_step = int(hp['decay_epoch'] * len(self.train_data))
        self.learning_rate = mu.get_lr(hp['decay_type'], hp['learning_rate'], self.global_step,
                                       decay_step, hp['decay_rate'])

        self.optimizer = mu.get_opt(hp['opt'], self.learning_rate, decay_step)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        gradients, variables = list(zip(*grads_and_vars))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = tf.group(self.optimizer.apply_gradients(grads_and_vars, self.global_step),
                                     self.train_acc_update,
                                     self.train_attn_acc_update)  # self.train_a_attn_acc_update, self.train_q_attn_acc_update)

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables())

        # Summary
        train_gv_summaries = []
        for idx, grad in enumerate(gradients):
            if grad is not None:
                train_gv_summaries.append(tf.summary.histogram('gradients/' + variables[idx].name, grad))
                train_gv_summaries.append(tf.summary.histogram(variables[idx].name, variables[idx]))

        train_summaries = [
            tf.summary.scalar('train_loss', self.loss),
            tf.summary.scalar('train_accuracy', self.train_acc),
            # tf.summary.scalar('train_a_attn_accuracy', self.train_a_attn_acc),
            # tf.summary.scalar('train_q_attn_accuracy', self.train_q_attn_acc),
            tf.summary.scalar('train_attn_accuracy', self.train_attn_acc),
            tf.summary.scalar('learning_rate', self.learning_rate)
        ]
        self.train_summaries_op = tf.summary.merge(train_summaries)
        self.train_gv_summaries_op = tf.summary.merge(train_gv_summaries + train_summaries)

        val_summaries = [
            tf.summary.scalar('val_accuracy', self.val_acc),
            tf.summary.scalar('val_attn_accuracy', self.val_attn_acc),
            # tf.summary.scalar('val_a_attn_accuracy', self.val_a_attn_acc),
            # tf.summary.scalar('val_q_attn_accuracy', self.val_q_attn_acc),
        ]
        self.val_summaries_op = tf.summary.merge(val_summaries)

        if args.checkpoint:
            self.checkpoint_file = args.checkpoint
        else:
            self.checkpoint_file = tf.train.latest_checkpoint(self._checkpoint_dir)

        self.train_init_op_list = [self.train_data.initializer, self.train_acc_init,
                                   # self.train_q_attn_acc_init, self.train_a_attn_acc_init,
                                   self.train_attn_acc_init]

        self.val_init_op_list = [self.val_data.initializer, self.val_acc_init,
                                 # self.val_q_attn_acc_init, self.val_a_attn_acc_init,
                                 self.val_attn_acc_init]

        self.train_op_list = [self.train_op, self.loss, self.attn_loss, self.train_acc,
                              self.train_attn_acc,  # self.val_q_attn_acc, self.val_a_attn_acc,
                              self.global_step, self.train_data.spec, self.train_attn]
        self.val_op_list = [self.val_acc, self.val_attn_acc,  # self.val_q_attn_acc, self.val_a_attn_acc,
                            tf.group(self.val_acc_update, self.val_attn_acc_update
                                     # self.val_q_attn_acc_update, self.val_a_attn_acc_update
                                     ),
                            self.val_summaries_op, self.val_data.spec, self.val_attn]
        # self.run_metadata = tf.RunMetadata()

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=config) as sess, tf.summary.FileWriter(self._log_dir, sess.graph) as sw:
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            if self.checkpoint_file:
                print('Restore from', self.checkpoint_file)
                self.saver.restore(sess, self.checkpoint_file)

            summary, acc, max_acc = None, 0, 0
            train_fig, val_fig = {}, {}
            try:
                while True:
                    step = tf.train.global_step(sess, self.global_step)
                    epoch = math.floor(step / len(self.train_data)) + 1

                    # Train Loop
                    sess.run(self.train_init_op_list, feed_dict=self.train_data.feed_dict)

                    with trange(epoch * len(self.train_data) - step) as pbar:

                        for i in pbar:
                            if step % 10000 == 0:
                                _, l, al, acc, a_acc, step, spec, attn, gv_summary = sess.run(
                                    self.train_op_list + [self.train_gv_summaries_op])
                                # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                # run_metadata=self.run_metadata)
                                # trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
                                sw.add_summary(gv_summary, step)
                                self.saver.save(sess, self._checkpoint_file, step)
                                # with open(args.mod + '.timeline.ctf.json', 'w') as trace_file:
                                #     trace_file.write(trace.generate_chrome_trace_format())
                            elif step % 100 == 0:
                                _, l, al, acc, a_acc, step, spec, attn, summary = sess.run(
                                    self.train_op_list + [self.train_summaries_op])
                                sw.add_summary(summary, step)
                            else:
                                _, l, al, acc, a_acc, step, spec, attn = sess.run(self.train_op_list)
                            pbar.set_description('[%03d] Train l: %.2f al: %.2f acc: %.2f a_acc: %.2f' %
                                                 (epoch, l, al, acc, a_acc))
                            qa = self.train_data.qa[self.train_data.index[i]]
                            train_fig[qa['qid'].replace(':', '')] = np.stack([spec, attn])

                    np.savez(os.path.join(self._attn_dir, 'train_attn.npz'), **train_fig)
                    self.saver.save(sess, self._checkpoint_file, step)

                    # Validation Loop
                    sess.run(self.val_init_op_list, feed_dict=self.val_data.feed_dict)
                    with trange(len(self.val_data)) as pbar:
                        for i in pbar:
                            acc, a_acc, _, summary, spec, attn = sess.run(self.val_op_list)
                            pbar.set_description(
                                '[%03d] Val acc: %.2f a_acc: %.2f' % (epoch, acc, a_acc))
                            qa = self.val_data.qa[self.val_data.index[i]]
                            val_fig[qa['qid'].replace(':', '')] = np.stack([spec, attn])
                        sw.add_summary(summary, step)
                        np.savez(os.path.join(self._attn_dir, 'val_attn.npz'), **val_fig)
                    if acc > max_acc:
                        max_acc = acc
                        self.best_saver.save(sess, self._best_checkpoint, step)
            except KeyboardInterrupt:
                self.saver.save(sess, self._checkpoint_file, self.global_step)

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
    parser.add_argument('--target', default='main/attn', help='Train what?')
    # parser.add_argument('--reg', action='store_true', help='Regularize the model.')
    args = parser.parse_args()
    mod = importlib.import_module('model.' + args.mod)
    hp = getattr(importlib.import_module('hp'), 'hp' + args.hp)
    reset = args.reset
    debug = args.debug
    target = args.target
    main()

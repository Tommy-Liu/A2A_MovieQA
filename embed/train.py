import math
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from glob import glob
from itertools import product
from os.path import join
from pprint import pprint

import tensorflow as tf

from data.tfrecord import TfRecordDataSet
from embed.args import EmbeddingPath, args_parse
from embed.model import NGramModel
from utils import data_utils as du
from utils import func_utils as fu
from utils.model_utils import get_loss, get_opt

cp = EmbeddingPath()


class EmbeddingData(object):
    def __init__(self, name, batch_size=128, num_threads=16):
        start_time = time.time()
        self.name = name
        self.records = TfRecordDataSet(OrderedDict([
            ('word', cp.encode_embedding_key_file),
            ('vec', cp.encode_embedding_vec_file)]
        ), name=self.name)

        self.batch_size = batch_size
        # Use floor instead of ceil because we drop last batch.
        self.total_step = int(math.ceil(self.records.num_example / self.batch_size))
        self.dataset = self.records.dataset \
            .shuffle(buffer_size=999) \
            .map(self.records.parse_fn,
                 num_threads=num_threads,
                 output_buffer_size=2048) \
            .shuffle(buffer_size=8192) \
            .batch(self.batch_size) \
            .repeat()
        self.iterator = self.dataset.make_initializable_iterator()
        self.word, self.vec = self.iterator.get_next()
        self.initializer = self.iterator.initializer
        self.vocab = du.json_load(cp.gram_vocab_file)
        self.vocab_size = len(self.vocab)
        print('Data Loading Finished with %.3f s.' % (time.time() - start_time))


class HyperParameter(object):
    def __init__(self, initial, progress, scale):
        self.value = initial
        if progress == 'arithmetic' or progress == 'arith' or progress == 'a':
            self.progress = partial(self.arith, scale)
            self.reverse = partial(self.rev_arith, initial, scale)
        elif progress == 'geometric' or progress == 'geo' or progress == 'g':
            self.progress = partial(self.geo, scale)
            self.reverse = partial(self.rev_geo, initial, scale)
        elif progress == 'manually' or progress == 'manu' or progress == 'm':
            self.progress = partial(self.manu, scale)
            self.reverse = partial(self.rev_manu, initial, scale)
        else:
            raise ValueError('Wrong progress value.')

    def __call__(self, i):
        return self.progress(self.value, i)

    def index(self, value):
        return self.reverse(value)

    @staticmethod
    def rev_arith(initial, scale, value):
        return round((value - initial) / scale)

    @staticmethod
    def rev_geo(initial, scale, value):
        return round(math.log(value / initial, scale))

    @staticmethod
    def rev_manu(initial, scale, value):
        return scale.index(initial) - scale.index(value)

    @staticmethod
    def arith(scale, value, i):
        return value + i * scale

    @staticmethod
    def geo(scale, value, i):
        return value * scale ** i

    @staticmethod
    def manu(scale, value, i):
        return scale[(scale.index(value) + i) % len(scale)]


class HyperParameterSelector(object):
    def __init__(self, hp, rest, parser, name):
        self.args = hp
        self.rest = rest
        self.parser = parser
        self.name = name
        self.target = OrderedDict(
            [('learning_rate', HyperParameter(self.parser.get_default('learning_rate'), 'g', 0.1)),
             ('initial_scale', HyperParameter(self.parser.get_default('initial_scale'), 'a', 0.0025)),
             ('initializer', HyperParameter(
                 self.parser.get_default('initializer'), 'm', ['glorot', 'random', 'orthogonal']))
             ])
        self.args['loss'] = 0
        self.args['lock'] = False
        self._select()

    @property
    def log_dir(self):
        return join(cp.log_dir, self.name, self.exp_name)

    @property
    def checkpoint_dir(self):
        return join(cp.checkpoint_dir, self.name, self.exp_name)

    @property
    def checkpoint_name(self):
        return join(self.checkpoint_dir, self.exp_name)

    @property
    def checkpoint_file(self):
        if self.rest['checkpoint_file']:
            return self.rest['checkpoint_file']
        else:
            return tf.train.latest_checkpoint(self.checkpoint_dir)

    @property
    def param_file(self):
        return join(self.log_dir, '%s.json' % self.exp_name)

    @property
    def exp_name(self):
        exp = []
        for idx, k in enumerate(self.args.keys()):
            if k not in ['loss', 'lock'] and \
                    not self._safe_equal(self.args[k], self.parser.get_default(k)) or \
                    k == 'learning_rate':
                exp.append('_'.join([k, fu.bb(self.args[k])]))
        return '&'.join(exp)

    @staticmethod
    def _safe_equal(a, b):
        if type(a) == type(b) and isinstance(a, float):
            return abs(a - b) < 10 ** (-8)
        else:
            return a == b

    def save(self):
        du.json_dump(self.args, self.param_file)

    @contextmanager
    def synchronization(self):
        try:
            self.args['lock'] = True
            self.save()
            yield
        finally:
            self.args['lock'] = False
            self.save()

    def setup(self):
        fu.make_dirs(self.log_dir)
        fu.make_dirs(self.checkpoint_dir)

    def _auto(self, experiments):
        for idx, d in enumerate(experiments):
            experiments[idx] = {k: d[k] for k in list(self.target.keys()) + ['lock', 'loss']}

        # tuple(0,0,...,0)
        grid = set(tuple(self.target[k].index(exp[k]) for k in self.target)
                   for exp in experiments)
        base = set(list(product(range(-1, 2), repeat=len(self.target))))
        chosen = list(iter(base.difference(grid)))[0]
        # change args' value
        for idx, k in enumerate(self.target.keys()):
            self.args[k] = self.target[k](chosen[idx])
        pprint(self.args)

    def _continue(self, experiments):
        for idx, d in enumerate(experiments):
            print('%d.' % (idx + 1))
            for k in self.args:
                if not self._safe_equal(d[k], self.args[k]) or k in ['lock', 'loss']:
                    print(k, d[k])

        while True:
            choice = input('Please, choose one:')
            if choice.isdecimal():
                choice = int(choice)
            else:
                print('Input is not decimal.')
                continue
            if 0 > choice > len(experiments):
                print('Index is out of range.')
                continue
            if experiments[choice - 1]['lock']:
                print('The experiment is ongoing.')
                continue
            break

        pprint(experiments[choice - 1])
        for k in experiments[choice - 1]:
            self.args[k] = experiments[choice - 1][k]

        if input('Reset? (yes -> True, else False)') == 'yes':
            fu.safe_remove(self.log_dir)
            fu.safe_remove(self.checkpoint_dir)

    def _select(self):
        if self.rest['auto'] or self.rest['continue']:
            exp_paths = glob(join(cp.log_dir, '**', '*.json'), recursive=True)
            experiments = []
            for p in exp_paths:
                experiments.append(du.json_load(p))
            if self.rest['auto']:
                self._auto(experiments)
            elif self.rest['continue']:
                self._continue(experiments)
        self.setup()


class EmbeddingTrainingManager(object):
    name = 'embedding'

    def __init__(self, hp, rest, parser):
        start_time = time.time()
        self.oracle = HyperParameterSelector(hp, rest, parser, self.name)
        with self.oracle.synchronization():
            self.data = EmbeddingData(self.name, self.oracle.args['batch_size'])
            self.model = NGramModel(self.data, self.oracle)
            self.loss = get_loss('cos', self.data, self.model)
            self.global_step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(
                self.oracle.args['learning_rate'], self.global_step,
                self.oracle.args['decay_epoch'] * self.data.total_step,
                self.oracle.args['decay_rate'], staircase=True)
            self.optimizer = get_opt(self.oracle.args['optimizer'], self.learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            # gradients, variables = list(zip(*grads_and_vars))
            self.train_op = self.optimizer.apply_gradients(grads_and_vars, self.global_step)
            self.saver = tf.train.Saver(tf.global_variables())
            self.summaries_op = tf.summary.merge(
                [tf.summary.scalar('loss', self.loss),
                 tf.summary.scalar('learning_rate', self.learning_rate)])
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            fu.block_print('Pipeline setup done with %.2fs' % (time.time() - start_time))

    def save(self, sess):
        self.saver.save(sess, self.oracle.checkpoint_name,
                        tf.train.global_step(sess, self.global_step))

    def train(self):
        start_time = time.time()
        config = tf.ConfigProto(allow_soft_placement=True, )
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sw = tf.summary.FileWriter(self.oracle.log_dir)
            # Initialize all variables
            self.init_op.run()
            if self.oracle.checkpoint_file:
                print('Restore from', self.oracle.checkpoint_file)
                self.saver.restore(sess, self.oracle.checkpoint_file)

            sess.run(self.data.initializer, feed_dict={
                self.data.records.file_names_placeholder: self.data.records.file_names})
            fu.block_print('Tensorflow initialization done with %.2fs' % (time.time() - start_time))
            step, vec, out = tf.train.global_step(sess, self.global_step), 'vec', 'out'
            try:
                while True:
                    if step % 9999 == 0:
                        ops = [self.loss, self.global_step, self.summaries_op, self.train_op,
                               self.data.vec, self.model.output]
                        loss, step, summary, _, vec, out = sess.run(ops)
                    else:
                        ops = [self.loss, self.global_step, self.summaries_op, self.train_op]
                        loss, step, summary, _ = sess.run(ops)

                    sw.add_summary(summary, tf.train.global_step(sess, self.global_step))
                    # Minimize loss of current length until loss unimproved.
                    if step % 10000 == 0:
                        self.save(sess)
                        pprint([vec, out])
                    if step % 1000 == 0:
                        print('\n'.join(['{:<20}: {:>20s}'.format(k, fu.bb(self.oracle.args[k]))
                                         for k in self.oracle.target]))
                        print('{:<20}: {:>20d}'.format('batch_size', self.data.batch_size))
                        print('{:<20}: {:>20d}'.format('total_step', self.data.total_step))
                        self.oracle.args['loss'] = loss
                    if step % 100 == 0:
                        print('epoch: {:02d} elapsed: {:>5.2f}s step: {:>6} loss: {:>10.5f}'
                              .format(step // self.data.total_step + 1,
                                      time.time() - start_time, step, loss))
                        start_time = time.time()
            except KeyboardInterrupt:
                self.save(sess)
            finally:
                sw.close()


def main():
    hp, rest, parser = args_parse()
    manager = EmbeddingTrainingManager(hp, rest, parser)
    with manager.oracle.synchronization():
        manager.train()


def test_data():
    data = EmbeddingData(EmbeddingTrainingManager.name, 8)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(data.initializer, feed_dict={data.records.file_names_placeholder: data.records.file_names})

        print(sess.run([data.vec, data.word]))


if __name__ == '__main__':
    main()

import argparse
import glob
import importlib
import os
import re
from collections import defaultdict, Counter

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import trange

from config import MovieQAPath
# from input import Input as In
# from input_v2 import Input as In2
from raw_input import TestInput
from utils import data_utils as du
from utils import func_utils as fu

_mp = MovieQAPath()


class TestManager(object):
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
        fu.make_dirs(_mp.test_dir)

        self.test_data = TestInput(mode=args.mode)

        self.test_model = mod.Model(self.test_data, scale=hp['reg'], training=True)

        self.test_answer = tf.argmax(self.test_model.output, axis=1)

        self.saver = tf.train.Saver(tf.global_variables())

        self.test_init_op_list = [self.test_data.initializer]

        # self.run_metadata = tf.RunMetadata()

    def test(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

        with tf.Session(config=config) as sess, tf.summary.FileWriter(self._log_dir, sess.graph) as sw:
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            if args.checkpoint:
                print('Restore from', args.checkpoint)
                self.saver.restore(sess, args.checkpoint)
            else:
                print('Restore from', tf.train.latest_checkpoint(self._checkpoint_dir))
                self.saver.restore(sess, tf.train.latest_checkpoint(self._checkpoint_dir))

            summary, acc, max_acc, test = None, 0, 0, {}
            sess.run(self.test_init_op_list, feed_dict=self.test_data.feed_dict)

            for i in trange(len(self.test_data)):
                ans = sess.run(self.test_answer)
                qid = self.test_data.qa[self.test_data.index[i]]['qid']
                test[qid] = int(ans[:])

            du.json_dump(test, self._test_file)

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

    @property
    def _test_file(self):
        return os.path.join(_mp.test_dir, self._model_name + '.json')


def main():
    tester = TestManager()
    try:
        for i in range(11, 21):
            args.extra = '%02d' % i
            tester.test()
        ans_file_list = glob.glob(os.path.join(_mp.test_dir, '*.json'))
        ans_dict = defaultdict(Counter)
        for f in ans_file_list:
            ans = du.json_load(f)
            for qid, a in ans.items():
                ans_dict[qid].update([a])

        with open('movie_results.txt', 'w') as f:
            for qid in sorted(ans_dict.keys(), key=lambda x: int(re.search(r'(\d+)\b', x)[0])):
                ta = ans_dict[qid].most_common(1)[0][0]
                f.write('%s %s\n' % (qid, ta))
    except KeyboardInterrupt:
        print()


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
    # parser.add_argument('--reg', action='store_true', help='Regularize the model.')
    args = parser.parse_args()
    mod = importlib.import_module('model.' + args.mod)
    hp = getattr(importlib.import_module('hp'), 'hp' + args.hp)
    reset = args.reset
    debug = args.debug
    attn = args.attn
    main()

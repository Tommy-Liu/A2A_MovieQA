import random
from functools import partial
from glob import glob

import numpy as np
import tensorflow as tf
from os.path import join

from config import MovieQAPath
from utils import data_utils as du

_mp = MovieQAPath()


def parse_feature():
    context_features = {
        "ques": tf.FixedLenFeature([300], dtype=tf.float32),
        "subt": tf.FixedLenFeature([], dtype=tf.string),
        "feat": tf.FixedLenFeature([], dtype=tf.string)
    }
    sequence_features = {
        "ans": tf.FixedLenSequenceFeature([300], dtype=tf.float32),
        # "spec": tf.FixedLenSequenceFeature([1], dtype=tf.int64),
    }

    return context_features, sequence_features


def feat_load(f):
    return np.load(f.decode('utf-8'))


def subt_load(f):
    subt = np.load(f.decode('utf-8'))
    return subt


def dual_parser(record, mode):
    context_features, sequence_features = parse_feature()

    context_features["gt"] = tf.FixedLenFeature([], dtype=tf.int64)

    c, s = tf.parse_single_sequence_example(record, context_features, sequence_features)

    res = [tf.expand_dims(c['ques'], axis=0), s['ans'],
           tf.expand_dims(c['gt'], axis=0)]

    if 'subt' in mode:
        subt = tf.py_func(subt_load, [c['subt']], [tf.float32])
        res.append(tf.reshape(subt, [-1, 300]))
    if 'feat' in mode:
        feat = tf.py_func(feat_load, [c['feat']], [tf.float32])
        res.append(tf.reshape(feat, [-1, 4, 4, 1536]))

    res.append(s['spec'])

    return res


def test_parser(record):
    context_features, sequence_features = parse_feature()
    c, s = tf.parse_single_sequence_example(record, context_features, sequence_features)

    return tf.expand_dims(c['ques'], axis=0), s['ans'], s['subt'], \
           tf.reshape(s['feat'], [-1, 64, 1536]), \
           tf.expand_dims(c['ql'], axis=0), c['al'], \
           tf.sparse_tensor_to_dense(c['sl']),


class TestInput(object):
    def __init__(self):
        self.test_pattern = join(_mp.dataset_dir, 'test*.tfrecord')
        self.test_files = glob(self.test_pattern)
        self.test_files.sort()
        self._length = len([0 for qa in du.json_load(_mp.qa_file) if 'test' in qa['qid'] and qa['video_clips']])
        dataset = tf.data.Dataset.from_tensor_slices(self.test_files)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1).prefetch(16)
        dataset = dataset.map(test_parser, num_parallel_calls=4).prefetch(16)
        self.iterator = dataset.make_initializable_iterator()
        self.next_elements = self.iterator.get_next()
        self.initializer = self.iterator.initializer
        self.ques, self.ans, self.subt, self.feat, self.ql, self.al, self.sl = self.next_elements

    def __len__(self):
        return self._length


class Input(object):
    def __init__(self, shuffle=True, split='train', mode='feat+subt'):
        self.shuffle = shuffle
        # if 'subt' not in mode:
        #     self.pattern = join(_mp.dataset_dir, '-'.join(['feat', split, '*.tfrecord']))
        # elif 'feat' not in mode:
        #     self.pattern = join(_mp.dataset_dir, '-'.join(['subt', split, '*.tfrecord']))
        # else:
        self.pattern = join(_mp.dataset_dir, split + '*.tfrecord')

        self._files = glob(self.pattern)
        self._files.sort()
        self._length = len([0 for qa in du.json_load(_mp.qa_file) if split in qa['qid'] and qa['video_clips']])

        parser = partial(dual_parser, mode=mode)

        self.placeholder = tf.placeholder(tf.string, [None])

        dataset = tf.data.Dataset.from_tensor_slices(self.placeholder)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1).prefetch(8)
        dataset = dataset.map(parser, num_parallel_calls=4).prefetch(8)
        if shuffle:
            dataset = dataset.shuffle(32)

        self.iterator = dataset.make_initializable_iterator()

        self.next_elements = self.iterator.get_next()

        self.initializer = self.iterator.initializer

        print(*[i.get_shape() for i in self.next_elements])

        if 'subt' not in mode:
            self.ques, self.ans, self.gt, self.feat, self.spec = self.next_elements
        elif 'feat' not in mode:
            self.ques, self.ans, self.gt, self.subt, self.spec = self.next_elements
        else:
            self.ques, self.ans, self.gt, self.subt, self.feat, self.spec = self.next_elements

    @property
    def files(self):
        if self.shuffle:
            random.shuffle(self._files)
        return self._files

    def __len__(self):
        return self._length


def main():
    train_data = Input(split='train', mode='subt')
    # foo_data = Input(split='val')
    # bar_data = TestInput()
    config = tf.ConfigProto(allow_soft_placement=True, )
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(train_data.initializer, feed_dict={train_data.placeholder: train_data.files})

        # for i in range(3):
        # for _ in trange(3, desc='Train loop'):
        e = sess.run(train_data.next_elements)
        print(e[0])
        s = e[-2][np.sum(e[-2], axis=1) != 0]
        print(s)
        print(s.shape)
        print(*[i.shape for i in e])

        # sess.run(foo_data.initializer)

        # for _ in trange(20, desc='Validation loop'):
        #     sess.run(foo_data.next_elements)
        # sess.run(foo_data.next_elements, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #          run_metadata=run_metadata)
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open('foo' + '.timeline.ctf.json', 'w') as trace_file:
        #     trace_file.write(trace.generate_chrome_trace_format())
        #
        # sess.run(bar_data.initializer, feed_dict={bar_data.placeholder: bar_data.files})
        #
        # for _ in trange(20, desc='Test loop'):
        #     sess.run(bar_data.next_elements)
        # sess.run(bar_data.next_elements, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #          run_metadata=run_metadata)
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open('bar' + '.timeline.ctf.json', 'w') as trace_file:
        #     trace_file.write(trace.generate_chrome_trace_format())


if __name__ == '__main__':
    main()

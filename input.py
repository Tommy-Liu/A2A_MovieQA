import random
from functools import partial
from glob import glob
from os.path import join

import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import trange

from config import MovieQAPath
from utils import data_utils as du

_mp = MovieQAPath()


def parse_feature():
    context_features = {
        "al": tf.FixedLenFeature([5], dtype=tf.int64),
        "ques": tf.FixedLenFeature([25], dtype=tf.int64),
        "ql": tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {'ans': tf.FixedLenSequenceFeature([34], dtype=tf.int64)}

    return context_features, sequence_features


def dual_parser(record, mode):
    context_features, sequence_features = parse_feature()
    if 'subt' in mode:
        context_features['sl'] = tf.VarLenFeature(dtype=tf.int64)
        sequence_features['subt'] = tf.FixedLenSequenceFeature([29], dtype=tf.int64)
    if 'feat' in mode:
        sequence_features['feat'] = tf.FixedLenSequenceFeature([8 * 8 * 1536], dtype=tf.float32)
    context_features["gt"] = tf.FixedLenFeature([], dtype=tf.int64)
    c, s = tf.parse_single_sequence_example(record, context_features, sequence_features)

    res = [tf.expand_dims(c['ques'], axis=0), s['ans'],
           tf.expand_dims(c['ql'], axis=0), c['al'], c['gt']]

    if 'subt' in mode:
        res.append(s['subt'])
        res.append(tf.sparse_tensor_to_dense(c['sl']))
    if 'feat' in mode:
        res.append(tf.reshape(s['feat'], [-1, 64, 1536]))

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
        if 'subt' not in mode:
            self.pattern = join(_mp.dataset_dir, '-'.join(['feat', split, '*.tfrecord']))
        elif 'feat' not in mode:
            self.pattern = join(_mp.dataset_dir, '-'.join(['subt', split, '*.tfrecord']))
        else:
            self.pattern = join(_mp.dataset_dir, split + '*.tfrecord')

        self.files = glob(self.pattern)
        if self.shuffle:
            random.shuffle(self.files)
        else:
            self.files.sort()
        self._length = len([0 for qa in du.json_load(_mp.qa_file) if split in qa['qid'] and qa['video_clips']])

        parser = partial(dual_parser, mode=mode)

        dataset = tf.data.Dataset.from_tensor_slices(self.files)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1).prefetch(8)
        dataset = dataset.map(parser, num_parallel_calls=4).prefetch(8)

        self.iterator = dataset.make_initializable_iterator()

        self.next_elements = self.iterator.get_next()

        self.initializer = self.iterator.initializer

        if 'subt' not in mode:
            self.ques, self.ans, self.ql, self.al, self.gt, self.feat = self.next_elements
        elif 'feat' not in mode:
            self.ques, self.ans, self.ql, self.al, self.gt, self.subt, self.sl = self.next_elements
        else:
            self.ques, self.ans, self.ql, self.al, self.gt, self.subt, self.sl, self.feat = self.next_elements

    def __len__(self):
        return self._length


def find_max_length(qa, subt):
    subt_max = 0
    for imdb_subt in subt.values():
        for v in imdb_subt.values():
            for sent in v:
                if subt_max < len(sent):
                    subt_max = len(sent)
    q_max, a_max = 0, 0
    for ins in qa:
        if q_max < len(ins['question']):
            q_max = len(ins['question'])
        for a in ins['answers']:
            if a_max < len(a):
                a_max = len(a)

    return subt_max, q_max, a_max


# options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
#                                     run_metadata=self.run_metadata

def main():
    train_data = Input(split='train')
    foo_data = Input(split='val')
    bar_data = TestInput()

    run_metadata = tf.RunMetadata()
    config = tf.ConfigProto(allow_soft_placement=True, )
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(train_data.initializer)

        # for i in range(3):
        for _ in trange(20, desc='Train loop'):
            sess.run(train_data.next_elements)
        sess.run(train_data.next_elements, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open('interleaves' + '.timeline.ctf.json', 'w') as trace_file:
            trace_file.write(trace.generate_chrome_trace_format())

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

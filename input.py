from glob import glob
from os.path import join
from random import shuffle

import tensorflow as tf

from config import MovieQAPath

_mp = MovieQAPath()


def parse_feature():
    context_features = {
        "sl": tf.VarLenFeature(dtype=tf.int64),
        "al": tf.FixedLenFeature([5], dtype=tf.int64),
        "ques": tf.FixedLenFeature([25], dtype=tf.int64),
        "ql": tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        "subt": tf.FixedLenSequenceFeature([29], dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([8 * 8 * 1536], dtype=tf.float32),
        "ans": tf.FixedLenSequenceFeature([34], dtype=tf.int64)
    }

    return context_features, sequence_features


def dual_parser(record):
    context_features, sequence_features = parse_feature()
    context_features["gt"] = tf.FixedLenFeature([], dtype=tf.int64)
    c, s = tf.parse_single_sequence_example(record, context_features, sequence_features)

    return tf.expand_dims(c['ques'], axis=0), s['ans'], s['subt'], \
           tf.reshape(s['feat'], [-1, 64, 1536]), \
           tf.expand_dims(c['ql'], axis=0), c['al'], \
           tf.sparse_tensor_to_dense(c['sl']), c['gt']


def test_parser(record):
    context_features, sequence_features = parse_feature()
    c, s = tf.parse_single_sequence_example(record, context_features, sequence_features)

    return tf.expand_dims(c['ques'], axis=0), s['ans'], s['subt'], \
           tf.reshape(s['feat'], [-1, 64, 1536]), \
           tf.expand_dims(c['ql'], axis=0), c['al'], \
           tf.sparse_tensor_to_dense(c['sl']),


class TestInput(object):
    def __init__(self):
        self.test_files = glob(join(_mp.dataset_dir, 'test*.tfrecord'))
        self.record_placeholder = tf.placeholder(tf.string, [None])
        dataset = tf.data.TFRecordDataset(self.record_placeholder) \
            .map(test_parser, num_parallel_calls=8).prefetch(16)
        self.iterator = dataset.make_initializable_iterator()
        self.next_elements = self.iterator.get_next()
        self.initializer = self.iterator.initializer
        self.ques, self.ans, self.subt, self.feat, self.ql, self.al, self.sl = \
            self.next_elements


class Input(object):
    def __init__(self, rand=True):
        self._train_files = glob(join(_mp.dataset_dir, 'train*.tfrecord'))
        self._val_files = glob(join(_mp.dataset_dir, 'val*.tfrecord'))
        self.shuffle = rand

        self.record_placeholder = tf.placeholder(tf.string, [None])

        dataset = tf.data.TFRecordDataset(self.record_placeholder) \
            .map(dual_parser, num_parallel_calls=8).prefetch(32)

        self.iterator = dataset.make_initializable_iterator()

        self.next_elements = self.iterator.get_next()

        self.initializer = self.iterator.initializer

        self.ques, self.ans, self.subt, self.feat, self.ql, self.al, self.sl, self.gt = \
            self.next_elements

    @property
    def train_files(self):
        if self.shuffle:
            shuffle(self._train_files)
        return self._train_files

    @property
    def val_files(self):
        if self.shuffle:
            shuffle(self._val_files)
        return self._val_files


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


def main():
    data = Input()

    test_data = TestInput()

    with tf.Session() as sess:
        sess.run(data.initializer, feed_dict={data.record_placeholder: data.train_files})

        # for i in range(3):
        print(sess.run(data.ques))

        sess.run(test_data.initializer, feed_dict={test_data.record_placeholder: test_data.test_files})

        # for i in range(3):
        print(sess.run(data.ques))


if __name__ == '__main__':
    main()

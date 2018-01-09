import math
import sys
import time
from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow.contrib.data as tf_data

from config import MovieQAConfig
from utils import data_utils as du

config = MovieQAConfig('..')


def feature_parser(record, embedding_size=300, max_length=12):
    features = {
        "vec": tf.FixedLenFeature([embedding_size], tf.float32),
        "word": tf.FixedLenFeature([max_length], tf.int64),
        "len": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, features)

    return parsed['vec'], parsed['word'], parsed['len']


class EmbeddingData(object):
    RECORD_FILE_PATTERN_ = join('embedding_dataset', 'embedding_%s*.tfrecord')

    def __init__(self, batch_size=128, num_thread=16, num_given=sys.maxsize,
                 use_length=12, raw_input=False, embedding_size=300, max_length=12):
        start_time = time.time()
        self.raw_len = np.load(config.encode_embedding_len_file)
        self.batch_size = batch_size
        # TODO(tommy8054): Raw input from numpy array. Is it necessary to implement this? QQ
        if raw_input:
            # Filter instances
            length = np.load(config.encode_embedding_len_file)
            vec = np.load(config.encode_embedding_vec_file)[length <= use_length]
            word = np.load(config.encode_embedding_key_file)[length <= use_length]
            length = length[length <= use_length]
            vec = vec[:min(len(vec), num_given)]
            word = word[:min(len(word), num_given)]
            length = length[:min(len(length), num_given)]
            # Build up input pipeline
            self.load = {'vec': vec, 'word': word, 'len': length}
            self.vec, self.word, self.len = \
                tf.placeholder(tf.float32, [None, embedding_size], 'vec'), \
                tf.placeholder(tf.int64, [None, max_length], 'word'), \
                tf.placeholder(tf.int64, [None], 'length')

            self.vec_temp, self.word_temp, self.len_temp = \
                tf.placeholder(tf.float32, [None, embedding_size], 'vec_temp'), \
                tf.placeholder(tf.int64, [None, max_length], 'word_temp'), \
                tf.placeholder(tf.int64, [None], 'length_temp')

            self.dataset = tf_data.Dataset.from_tensor_slices(
                (self.vec_temp, self.word_temp, self.len_temp)) \
                .prefetch(num_thread * batch_size * 4) \
                .shuffle(buffer_size=num_thread * batch_size * 8) \
                .apply(tf_data.batch_and_drop_remainder(batch_size))
            self.iterator = self.dataset.make_initializable_iterator()
            self.input = self.iterator.get_next()
        else:
            self.num_each_len = [np.sum(self.raw_len == (i + 1), dtype=np.int64) for i in range(max_length)]
            self.num_example = min(len(self.raw_len), num_given)
            # Use floor instead of ceil because we drop last batch.
            self.total_step = int(math.floor(self.num_example / self.batch_size))
            self.file_names = glob(self.RECORD_FILE_PATTERN_ % 'length_')
            self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
            self.dataset = tf.data.TFRecordDataset(self.file_names_placeholder) \
                .shuffle(buffer_size=160) \
                .map(feature_parser, num_parallel_calls=num_thread) \
                .prefetch(2000) \
                .shuffle(buffer_size=1000) \
                .apply(tf_data.batch_and_drop_remainder(batch_size)) \
                .repeat()
            self.iterator = self.dataset.make_initializable_iterator()
            self.vec, self.word, self.len = self.iterator.get_next()
        self.vocab = du.json_load(config.char_vocab_file)
        self.vocab_size = len(self.vocab)
        print('Data Loading Finished with %.3f s.' % (time.time() - start_time))

    def test(self):
        with tf.Session() as sess:
            sess.run(self.iterator.initializer, feed_dict={
                self.file_names_placeholder: self.file_names
            })

            print(sess.run([self.vec, self.word, self.len]))

    def get_records(self, length):
        self.num_example = sum([self.num_each_len[i - 1] for i in length])
        self.total_step = int(math.floor(self.num_example / self.batch_size))
        return [n for n in self.file_names for l in length if 'length_%d_' % l in n]

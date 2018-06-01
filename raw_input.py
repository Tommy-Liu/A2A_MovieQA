import random
from functools import partial
from os.path import join

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import MovieQAPath
from utils import data_utils as du

_mp = MovieQAPath()
embedding_size = 300


def subt_load(s, mode):
    if 'subt' in mode:
        return np.load(s.decode('utf-8')).astype(np.float32)
    else:
        return np.zeros((1, embedding_size), dtype=np.float32)


def feat_load(f, mode):
    if 'feat' in mode:
        return np.load(f.decode('utf-8')).astype(np.float32)
    else:
        return np.zeros((1, 6, 2048), dtype=np.float32)


def qa_load(qa):
    qa = np.load(qa.decode('utf-8')).astype(np.float32)
    return qa[:1], qa[1:6]


def spec_load(spec):
    return np.load(spec.decode('utf-8')).astype(np.int32)


def load(tensor, comp, mode):
    if comp == 'qa':
        func = qa_load
        q, a = tf.py_func(func, [tensor], [tf.float32, tf.float32])
        return tf.reshape(q, [-1, embedding_size]), tf.reshape(a, [-1, embedding_size])
    elif comp == 'subt':
        func = partial(subt_load, mode=mode)
        return tf.reshape(tf.py_func(func, [tensor], [tf.float32]), [-1, embedding_size])
    elif comp == 'spec':
        func = spec_load
        # return tf.reshape(tf.py_func(func, [tensor], [tf.int64]), [-1, 1])
        return tf.reshape(tf.py_func(func, [tensor], [tf.int32]), [-1])
    else:
        func = partial(feat_load, mode=mode)
        return tf.reshape(tf.py_func(func, [tensor], [tf.float32]), [-1, 6, 2048])


class Input(object):
    def __init__(self, split='train', mode='feat+subt', shuffle=True):
        self.shuffle = shuffle
        vsqa = [qa for qa in du.json_load(_mp.qa_file) if qa['video_clips']]
        self.qa = [qa for qa in vsqa if split in qa['qid']]
        self.index = list(range(len(self)))
        self._feed_dict = {
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.encode_dir, qa['qid'] + '.npy') for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.encode_dir, qa['imdb_key'] + '.npy') for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.object_feature_dir, qa['imdb_key'] + '.npy') for qa in self.qa],
            tf.placeholder(dtype=tf.int64, shape=[None]): [qa['correct_index'] for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.encode_dir, qa['qid'] + '_spec' + '.npy') for qa in self.qa],
        }
        self.placeholders = list(self.feed_dict.keys())
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[0]).repeat(1)
        func = partial(load, comp='qa', mode=mode)
        qa_dataset = dataset.map(func, num_parallel_calls=1).prefetch(1)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[1]).repeat(1)
        func = partial(load, comp='subt', mode=mode)
        subt_dataset = dataset.map(func, num_parallel_calls=2).prefetch(2)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[2]).repeat(1)
        func = partial(load, comp='feat', mode=mode)
        feat_dataset = dataset.map(func, num_parallel_calls=4).prefetch(4)
        gt_dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[3]).repeat(1)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[4]).repeat(1)
        func = partial(load, comp='spec', mode=mode)
        spec_dataset = dataset.map(func, num_parallel_calls=1).prefetch(1)

        dataset = tf.data.Dataset.zip((qa_dataset, subt_dataset, feat_dataset, gt_dataset, spec_dataset))
        dataset = dataset.prefetch(128)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        (self.ques, self.ans), self.subt, self.feat, self.gt, self.spec = next_element
        self.gt = tf.expand_dims(self.gt, axis=0)
        self.next_element = (self.ques, self.ans, self.subt, self.feat, self.gt, self.spec)
        self.initializer = iterator.initializer

    def __len__(self):
        return len(self.qa)

    @property
    def feed_dict(self):
        if self.shuffle:
            random.shuffle(self.index)
            return {k: [self._feed_dict[k][i] for i in self.index] for k in self._feed_dict}
        else:
            return self._feed_dict


class TestInput(object):
    def __init__(self, mode='feat+subt', shuffle=True):
        self.shuffle = shuffle
        vsqa = [qa for qa in du.json_load(_mp.qa_file) if qa['video_clips']]
        self.qa = [qa for qa in vsqa if 'test' in qa['qid']]
        self.index = list(range(len(self)))
        self._feed_dict = {
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.encode_dir, qa['qid'] + '.npy') for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.encode_dir, qa['imdb_key'] + '.npy') for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.object_feature_dir, qa['imdb_key'] + '.npy') for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [join(_mp.encode_dir, qa['qid'] + '_spec' + '.npy') for qa in self.qa],
        }
        self.placeholders = list(self.feed_dict.keys())
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[0]).repeat(1)
        func = partial(load, comp='qa', mode=mode)
        qa_dataset = dataset.map(func, num_parallel_calls=1).prefetch(1)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[1]).repeat(1)
        func = partial(load, comp='subt', mode=mode)
        subt_dataset = dataset.map(func, num_parallel_calls=2).prefetch(2)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[2]).repeat(1)
        func = partial(load, comp='feat', mode=mode)
        feat_dataset = dataset.map(func, num_parallel_calls=4).prefetch(4)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[3]).repeat(1)
        func = partial(load, comp='spec', mode=mode)
        spec_dataset = dataset.map(func, num_parallel_calls=1).prefetch(1)

        dataset = tf.data.Dataset.zip((qa_dataset, subt_dataset, feat_dataset, spec_dataset))
        dataset = dataset.prefetch(128)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        (self.ques, self.ans), self.subt, self.feat, self.spec = next_element
        self.next_element = (self.ques, self.ans, self.subt, self.feat, self.spec)
        self.initializer = iterator.initializer

    def __len__(self):
        return len(self.qa)

    @property
    def feed_dict(self):
        if self.shuffle:
            random.shuffle(self.index)
            return {k: [self._feed_dict[k][i] for i in self.index] for k in self._feed_dict}
        else:
            return self._feed_dict


def main():
    # data = Input(mode='subt+feat')
    # data2 = Input(mode='subt+feat', split='val')
    data2 = TestInput(shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # sess.run([data.initializer, data2.initializer], feed_dict={**data.feed_dict, **data2.feed_dict})
        sess.run([data2.initializer], feed_dict={**data2.feed_dict})
        # np.set_printoptions(threshold=np.inf)
        # for _ in trange(len(data.qa)):
        #     f, s = sess.run([data.feat, data.subt])
        #     # print(q1, q2)
        #     # print(q1.shape, q2.shape)
        #     assert f.shape[0] == s.shape[0], 'Shit'
        for _ in trange(len(data2.qa)):
            f, s = sess.run([data2.feat, data2.subt])
            # print(q1, q2)
            # print(q1.shape, q2.shape)
            assert f.shape[0] == s.shape[0], 'Shit'


if __name__ == '__main__':
    main()

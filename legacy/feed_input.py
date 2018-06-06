import random
from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import trange

from config import MovieQAPath
from utils import data_utils as du

_mp = MovieQAPath()
embedding_size = 300


def _qa_load(s, qa):
    qid = s.decode('utf-8')
    enc = qa[qid].astype(np.float32)
    spec = qa[qid + 'spec'].astype(np.int32)
    gt = qa[qid + 'correct_index'].astype(np.int64)
    return enc[:1], enc[1:6], spec, gt


def qa_load(tensor, qa):
    func = partial(_qa_load, qa=qa)
    q, a, spec, gt = tf.py_func(func, [tensor], [tf.float32, tf.float32, tf.int32, tf.int64])
    return tf.reshape(q, [-1, embedding_size]), tf.reshape(a, [-1, embedding_size]), \
           tf.reshape(spec, [-1]), tf.reshape(gt, [-1])


def _feat_load(s, feature, subtitle, mode):
    imdb_key = s.decode('utf-8')
    if 'subt' in mode and 'feat' in mode:
        return subtitle[imdb_key].astype(np.float32), feature[imdb_key].astype(np.float32)
    elif 'subt' in mode:
        return subtitle[imdb_key].astype(np.float32), np.zeros((1, 6, 2048), dtype=np.float32)
    elif 'feat' in mode:
        return np.zeros((1, embedding_size), dtype=np.float32), feature[imdb_key].astype(np.float32)
    else:
        return np.zeros((1, embedding_size), dtype=np.float32), np.zeros((1, 6, 2048), dtype=np.float32)


def feat_load(tensor, feature, subtitle, mode):
    func = partial(_feat_load, feature=feature, subtitle=subtitle, mode=mode)
    subt, feat = tf.py_func(func, [tensor], [tf.float32, tf.float32])
    return tf.reshape(subt, [-1, embedding_size]), tf.reshape(feat, [-1, 6, 2048])


class Input(object):
    def __init__(self, split='train', mode='feat+subt', shuffle=True):
        self.shuffle = shuffle
        vsqa = [qa for qa in du.json_load(_mp.qa_file) if qa['video_clips']]
        self.qa = [qa for qa in vsqa if split in qa['qid']]
        self.index = list(range(len(self)))
        self.feature = dict(np.load(_mp.object_feature))
        self.subtitle = dict(np.load(_mp.subtitle_feature))
        self.qa_feature = dict(np.load(_mp.qa_feature))
        self._feed_dict = {
            tf.placeholder(dtype=tf.string, shape=[None]):
                [qa['qid'] for qa in self.qa],
            tf.placeholder(dtype=tf.string, shape=[None]):
                [qa['imdb_key'] for qa in self.qa]
        }
        self.placeholders = list(self.feed_dict.keys())

        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[0]).repeat(1)
        func = partial(qa_load, qa=self.qa_feature)
        qa_dataset = dataset.map(func, num_parallel_calls=2).prefetch(2)
        dataset = tf.data.Dataset.from_tensor_slices(self.placeholders[1]).repeat(1)
        func = partial(feat_load, feature=self.feature, subtitle=self.subtitle, mode=mode)
        feat_dataset = dataset.map(func, num_parallel_calls=4).prefetch(4)

        dataset = tf.data.Dataset.zip((qa_dataset, feat_dataset))
        dataset = dataset.prefetch(128)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        (self.ques, self.ans, self.spec, self.gt), (self.subt, self.feat) = next_element
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


def main():
    data = Input(mode='subt+feat')
    data2 = Input(mode='subt+feat', split='val')

    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run([data.initializer, data2.initializer], feed_dict={**data.feed_dict, **data2.feed_dict})
        # np.set_printoptions(threshold=np.inf)
        for _ in trange(len(data.qa)):
            f, s = sess.run([data.feat, data.subt])
            # print(q1, q2)
            # print(q1.shape, q2.shape)
            assert f.shape[0] == s.shape[0], 'Shit'
        for _ in trange(len(data2.qa)):
            f, s = sess.run([data2.feat, data2.subt])
            # print(q1, q2)
            # print(q1.shape, q2.shape)
            assert f.shape[0] == s.shape[0], 'Shit'


if __name__ == '__main__':
    main()

# import argparse
import string
import math
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join, exists

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from tqdm import trange, tqdm

from utils import data_utils as du
from utils import func_utils as fu


def create_one_tfrecord(pattern, num_shards, ex_tuple):
    shard_id, length, feature_dict = ex_tuple
    output_filename = pattern % (shard_id + 1, num_shards)
    fu.exist_then_remove(output_filename)

    def create_sequence_example(d, idx):
        return tf.train.SequenceExample(
            context=tf.train.Features(feature={k: du.to_feature(d['features'][k][idx])
                                               for k in d['features'].keys()}),
            feature_lists=tf.train.FeatureLists(feature_list={k: du.to_feature(d['feature_list'][k][idx])
                                                              for k in d['feature_list'].keys()}))

    def create_example(d, idx):
        return tf.train.Example(
            features=tf.train.Features(feature={k: du.to_feature(d['featurs'][k][idx])
                                                for k in d['features'].keys()}))

    example_func = create_example if feature_dict['feature_list'] else create_sequence_example
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(length):
            example = example_func(feature_dict, i)
            tfrecord_writer.write(example.SerializeToString())


class TfRecordDataSet(object):
    TFRECORD_PATTERN = '_%05d-of-%05d.tfrecord'
    TFRECORD_PATTERN_ = '*.tfrecord'
    info_file = 'dataset_info.json'
    dset_dir = './dataset'

    def __init__(self, target=None, dset_name='default', num_shards=128, num_threads=8):
        """Target is a dictionary containing data."""
        if exists(self.info_file):
            self.info = du.jload(self.info_file)
        self.dset_name = dset_name
        fu.exist_make_dirs(self.dset_dir)
        self.num_shards = num_shards
        self.num_threads = num_threads
        self.num_example = 0
        if target:
            self.create_tfrecord(target)
        self.file_names = glob(join(self.dset_dir, dset_name + self.TFRECORD_PATTERN_))
        if not target:
            self.num_example = self.get_example_num()
        self.file_names_placeholder = tf.placeholder(tf.string, [None])
        self.dataset = TFRecordDataset(self.file_names_placeholder)

    def get_example_num(self):
        num_examples = 0
        for tfrecord_file in tqdm(self.file_names):
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                num_examples += 1
        return num_examples

    def create_tfrecord(self, target):
        self.num_example, *check_tail = set(len(v) for v in target.values())
        assert len(check_tail) > 0, 'Different length of targets.'
        num_per_shard = int(math.ceil(self.num_example / float(self.num_shards)))

        features = {}  # Example
        context_features = {}  # Parse
        feature_list = {}  # Example
        sequence_features = {}  # Parse

        for k in target.keys():
            # Iterate over targets
            t = target[k]
            if isinstance(t, np.ndarray):
                # t is numpy array
                dim = t.ndim
                # decide whether t is a sequence or not (dim == 3 )
                if dim == 3:
                    feature_list[k] = t
                    sequence_features[k] = tf.FixedLenSequenceFeature(
                        shape=[t.shape[dim - 1]], dtype=tf.as_dtype(t.dtype))

                elif 0 < dim < 3:
                    features[k] = target[k]
                    context_features[k] = tf.FixedLenFeature(
                        shape=[t.shape[dim - 1]] if dim == 2 else [], dtype=tf.as_dtype(t.dtype)
                    )
                else:
                    raise ValueError('Wrong dimension (%d) of target value. Can\'t be processed later.' % dim)
            else:
                def depth(l):
                    return isinstance(l, list) and max(map(depth, l), default=0) + 1

                def varlen(l):
                    return any(map(varlen, l)) if depth(l) > 2 else len(set(map(len, l))) - 1

                if depth(t) == 3:
                    feature_list[k] = t
                    if varlen(t):
                        sequence_features[k] = tf.VarLenFeature(dtype=tf.as_dtype())
                elif 0 < depth(target[k]) < 3:
                    features[k] = target[k]
                else:
                    raise ValueError('Wrong depth (%d) of target value. Can\'t be processed later.' % depth(t))

        example_list = []
        for j in trange(self.num_shards):
            start_ndx = j * num_per_shard
            end_ndx = min((j + 1) * num_per_shard, self.num_example)
            example_list.append((j, end_ndx - start_ndx, {
                'features': {
                    k: features[k][start_ndx:end_ndx] for k in features.keys()
                },
                'feature_list': {
                    k: features[k][start_ndx:end_ndx] for k in feature_list.keys()
                }
            }))
        func = partial(create_one_tfrecord, self.TFRECORD_PATTERN, self.num_shards)
        with Pool(self.num_threads) as pool, tqdm(total=self.num_shards, desc=self.dset_name) as pbar:
            for _ in pool.imap_unordered(func, example_list):
                pbar.update()


if __name__ == '__main__':
    test_data = {
        # int list
        'a': list(range(20)),  # 20 x 1
        'b': [list(range(20)) for _ in range(20)],  # 20 x 20
        'c': [[list(range(20)) for _ in range(20)] for _ in range(20)],  # 20 x 20 x 20
        # float list
        'd': [0.1 for _ in range(20)],  # 20 x 1
        'e': [[0.1 for _ in range(20)]for _ in range(20)],  # 20 x 20
        'f': [[[0.1 for _ in range(20)]for _ in range(20)]for _ in range(20)],  # 20 x 20 x 20
        # string list
        'g': [[str.encode('gg') for _ in range(20)]for _ in range(20)],  # 20 x 20 x 2
        'h': [[[str.encode('gg') for _ in range(20)]for _ in range(20)]for _ in range(20)],  # 20 x 20 x 20 x 2
        # int numpy array
        'i': np.ones(20, dtype=np.int64),  # 20 x 1
        'j': np.ones((20, 20), dtype=np.int64),  # 20 x 20
        'k': np.ones((20, 20, 20), dtype=np.int64),  # 20 x 20 x 20
        # float numpy array
        'l': np.ones(20, dtype=np.float32),  # 20 x 1
        'm': np.ones((20, 20), dtype=np.float32),  # 20 x 20
        'n': np.ones((20, 20, 20), dtype=np.float32),  # 20 x 20 x 20
    }
    # data = TFRecordDataset(test_data)
    a = tf.train.Feature(int64_list=tf.train.Int64List(value=np.zeros(10, dtype=np.int64)))
    b = tf.train.Features(feature={'b': du.to_feature(test_data['j'][0])})
    c = tf.train.FeatureLists(feature_list={'c': du.to_feature(test_data['k'][0])})
    print(a, b, c)
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

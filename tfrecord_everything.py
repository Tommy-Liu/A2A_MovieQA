import math
from collections import OrderedDict
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join
from pprint import pprint

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from embed.args import CommonParameter
from utils import data_utils as du
from utils import func_utils as fu


def create_one_tfrecord(pack):
    fu.safe_remove(pack.record_name)
    with tf.python_io.TFRecordWriter(pack.record_name) as tfrecord_writer:
        for example in pack:
            tfrecord_writer.write(example.SerializeToString())


class Sender(object):
    """Sender determines the content of a package.
    Each package contains a Sender which takes care of distributing data to an example.
    If your data is very complicated, this class would not be suitable. Please overwrite
    this class to your own version."""

    def __init__(self, value):
        self.value = value
        self.index = 0

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        self.index = 0
        return self

    def __getitem__(self, index):
        return self.value[index]

    def __next__(self):
        if self.index < len(self):
            self.index += 1
            return self[self.index - 1]
        else:
            raise StopIteration


# TODO(tommy8054): Consider using threading to prefetch data and share them by SimpleQueue().
class MailPerson(object):
    """Deliver package to each worker.
    In our setting, we give exclusive subset of data to each worker. For small data,
    it is ok. However, if data grows much larger, we should consider this class as a
    producer to prefetch data and deliver to worker by queue."""

    def __init__(self, pattern, num_shards, num_example, features, feature_list):
        self.pattern = pattern
        self.num_shards = num_shards
        self.num_example = num_example
        self.num_per_shard = int(math.ceil(self.num_example / float(self.num_shards)))
        self.features = features
        self.feature_list = feature_list
        self.index = 0

    def __len__(self):
        return self.num_shards

    def __iter__(self):
        return self

    def __getitem__(self, index):
        start_ndx = index * self.num_per_shard
        end_ndx = min((index + 1) * self.num_per_shard, self.num_example)
        return Package(self.pattern % (index + 1, self.num_shards),
                       end_ndx - start_ndx,
                       {k: v[start_ndx:end_ndx] for k, v in self.features.items()},
                       {k: v[start_ndx:end_ndx] for k, v in self.feature_list.items()}, )

    def __next__(self):
        if self.index < self.num_shards:
            self.index += 1
            return self[self.index - 1]
        else:
            raise StopIteration


class Package(object):
    """Package contain items for worker to work on."""

    def __init__(self, record_name, length, features, feature_list):
        assert len(set(
            len(v) for v in list(feature_list.values()) + list(features.values()))) \
               == 1, 'Numbers of first dimension in data are mismatched.'
        self.record_name = record_name
        self.length = length
        self.features = features
        self.feature_list = feature_list
        self.index = 0

    def _create_example(self, index):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    k: du.to_feature(v[index])
                    for k, v in self.features.items()
                }))
        # Iterate over feature and get one of instance in (key, value) pairs.

    def _create_sequence_example(self, index):
        return tf.train.SequenceExample(
            context=tf.train.Features(feature={
                k: du.to_feature(v[index])
                for k, v in self.features.items()
            }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    k: du.to_feature(v[index])
                    for k, v in self.feature_list.items()}))

    def __getitem__(self, index):
        if self.feature_list:
            return self._create_sequence_example(index)
        else:
            return self._create_example(index)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.length:
            self.index += 1
            return self[self.index - 1]
        else:
            raise StopIteration


class TfRecordDataSet(object):
    TFRECORD_PATTERN_ = '*.tfrecord'
    root_dir = './dataset'

    # The dir of datset is at ./dataset/$name
    def __init__(self, target=None, name='default', num_shards=128, num_threads=8):
        """Target is a dictionary containing data (which can only be numpy array).
        Caution! Input target should be an OrderedDict, because you should know the
        order of keys for later data output"""
        self.num_shards = num_shards
        self.num_threads = num_threads
        self.target = target

        # Directory setup
        self.target_dir = join(self.root_dir, name)  # ./dataset/$name
        fu.make_dirs(self.target_dir)

        # File name setup
        self.tfrecord_pattern = join(self.target_dir, name + '_%05d-of-%05d.tfrecord')
        self.file_names = glob(join(self.target_dir, self.TFRECORD_PATTERN_))
        self.info_file = join(self.target_dir, 'dataset_info.json')

        # Feature dictionary for later parsing.
        self.context_features = OrderedDict()
        self.sequence_features = OrderedDict()
        self.parse_fn = None

        # Info contains number of examples and parse information in dataset.
        self.info = du.exist_json_load(self.info_file)
        pprint(self.info, indent=4)

        # If given target, create tfrecords.
        if target and not self.verify_tfrecord():
            print('Creating tfrecords....')
            self.create_tfrecord()
        elif not self.info or not self.file_names:
            raise ValueError('Target dictionary is none, and there is no available dataset')

        # Prepare feature dictionary.
        self.prepare_features()

        # Number of example
        self.num_example = self.info['number_example'] \
            if self.info.get('number_example', None) \
            else self.get_example_num()

        # Tensorflow dataset pipeline
        self.file_names_placeholder = tf.placeholder(tf.string, [None])
        self.dataset = tf.contrib.data.TFRecordDataset(self.file_names_placeholder)

    @staticmethod
    def parse_single_example(features, order, record):
        parsed = tf.parse_single_example(record, features)
        return tuple(parsed[k] for k in order)

    @staticmethod
    def parse_single_sequence_example(context_features, sequence_features, order, record):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            record, context_features, sequence_features)
        return tuple(context_parsed.get(k, None) or sequence_parsed.get(k, None) for k in order)

    def verify_tfrecord(self):
        if self.info.get('num_shards', None):
            return self.info['num_shards'] == len(self.file_names)
        else:
            return False

    def get_example_num(self):
        num_examples = 0
        for tfrecord_file in tqdm(self.file_names):
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                num_examples += 1
        return num_examples

    def prepare_features(self):
        for k in self.info['data'].keys():
            t = self.info['data'][k]
            dim = t['dim']
            shape = t['shape']
            dtype = t['dtype']
            if dim == 3:
                self.sequence_features[k] = tf.FixedLenSequenceFeature(
                    shape=[shape[dim - 1]], dtype=tf.as_dtype(dtype))
            elif 0 < dim < 3:
                self.context_features[k] = tf.FixedLenFeature(
                    shape=[shape[dim - 1]] if dim == 2 else [],  # Shape is last dim of tensor or just a scalar
                    dtype=tf.as_dtype(dtype))
            else:
                raise ValueError('Wrong dimension (%d) of target value. Can\'t be processed later.' % dim)
        pprint(self.context_features, indent=4)
        if self.sequence_features:
            self.parse_fn = partial(self.parse_single_sequence_example,
                                    self.context_features, self.sequence_features, list(self.target.keys()))
        else:
            self.parse_fn = partial(self.parse_single_example,
                                    self.context_features, list(self.target.keys()))

    def create_tfrecord(self):
        target = {k: np.load(v) for k, v in self.target.items()}
        self.num_example, *check_tail = list(set(len(v) for v in target.values()))
        assert len(check_tail) == 0, 'Different length of targets. %s' % check_tail
        self.info['number_example'] = self.num_example
        self.info['num_shards'] = self.num_shards
        self.info['data'] = {}
        features = {}  # Example
        feature_list = {}  # Example

        for k in target.keys():
            # Iterate over targets
            t = target[k]
            # Save tensor information
            self.info['data'][k] = {
                'dim': t.ndim,
                'shape': t.shape,
                'dtype': str(t.dtype)
            }
            # t is numpy array
            dim = t.ndim
            # decide whether t is a sequence or not (dim == 3 )
            if dim == 3:
                feature_list[k] = t
            elif 0 < dim < 3:
                features[k] = t
            else:
                raise ValueError('Wrong dimension (%d) of target value. Can\'t be processed later.' % dim)
            # else:
            #     def depth(l):
            #         return isinstance(l, list) and max(map(depth, l), default=0) + 1
            #
            #     def varlen(l):
            #         return any(map(varlen, l)) if depth(l) > 2 else len(set(map(len, l))) - 1
            #
            #     if depth(t) == 3:
            #         feature_list[k] = t
            #         if varlen(t):
            #             sequence_features[k] = tf.VarLenFeature(dtype=tf.as_dtype(du.probe_type(t)))
            #     elif 0 < depth(t) < 3:
            #         features[k] = t
            #         context_features = tf.FixedLenFeature(shape=[], dtype=tf.as_dtype(du.probe_type(t)))
            #     else:
            #         raise ValueError('Wrong depth (%d) of target value. Can\'t be processed later.' % depth(t))
        du.json_dump(self.info, self.info_file)
        deliverer = MailPerson(self.tfrecord_pattern, self.num_shards, self.num_example,
                               features, feature_list)

        with Pool(self.num_threads) as pool, tqdm(total=self.num_shards, desc=self.target_dir) as pbar:
            for _ in pool.imap_unordered(create_one_tfrecord, deliverer):
                pbar.update()


if __name__ == '__main__':
    cp = CommonParameter()
    d = {'vec': cp.encode_embedding_vec_file,
         'word': cp.encode_embedding_key_file}
    data = TfRecordDataSet(name='embedding')
    # test_data = {
    #     # int list
    #     'a': list(range(20)),  # 20 x 1
    #     'b': [list(range(20)) for _ in range(20)],  # 20 x 20
    #     'c': [[list(range(20)) for _ in range(20)] for _ in range(20)],  # 20 x 20 x 20
    #     # float list
    #     'd': [0.1 for _ in range(20)],  # 20 x 1
    #     'e': [[0.1 for _ in range(20)] for _ in range(20)],  # 20 x 20
    #     'f': [[[0.1 for _ in range(20)] for _ in range(20)] for _ in range(20)],  # 20 x 20 x 20
    #     # string list
    #     'g': [[str.encode('gg') for _ in range(20)] for _ in range(20)],  # 20 x 20 x 2
    #     'h': [[[str.encode('gg') for _ in range(20)] for _ in range(20)] for _ in range(20)],  # 20 x 20 x 20 x 2
    #     # int numpy array
    #     'i': np.ones(20, dtype=np.int64),  # 20 x 1
    #     'j': np.ones((20, 20), dtype=np.int64),  # 20 x 20
    #     'k': np.ones((20, 20, 20), dtype=np.int64),  # 20 x 20 x 20
    #     # float numpy array
    #     'l': np.ones(20, dtype=np.float32),  # 20 x 1
    #     'm': np.ones((20, 20), dtype=np.float32),  # 20 x 20
    #     'n': np.ones((20, 20, 20), dtype=np.float32),  # 20 x 20 x 20
    # }
    # data = TFRecordDataset(test_data)
    # a = tf.train.Features(feature={'a': du.to_feature(test_data['i'][0])})
    # b = tf.train.Features(feature={'b': du.to_feature(test_data['j'][0])})
    # c = tf.train.FeatureLists(feature_list={'c': du.to_feature(test_data['k'][0])})
    # print(a, b, c)
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

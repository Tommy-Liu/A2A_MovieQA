# import argparse
import math
import numpy as np
import tensorflow as tf
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join, exists
from tensorflow.contrib.data import TFRecordDataset
from tqdm import trange, tqdm

import data_utils as du


def create_one_tfrecord(pattern, num_shards, ex_tuple):
    shard_id, length, feature_dict = ex_tuple
    output_filename = pattern % (shard_id + 1, num_shards)
    du.exist_then_remove(output_filename)

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
        if exists(self.info_file):
            self.info = du.load_json(self.info_file)
        self.dset_name = dset_name
        du.exist_make_dirs(self.dset_dir)
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

        features = {}
        context_features = {}
        feature_list = {}
        sequence_features = {}

        for k in target.keys():
            t = target[k]
            if isinstance(t, np.ndarray):
                dim = t.ndim
                if dim == 3:
                    feature_list[k] = t
                    sequence_features[k] = tf.FixedLenSequenceFeature(
                        shape=[t.shape[dim - 1]], dtype=tf.as_dtype(t.dtype)
                    )

                elif 0 < dim < 3:
                    features[k] = target[k]
                    context_features[k] = tf.FixedLenFeature(
                        shape=[t.shape[dim - 1]] if dim == 2 else [], dtype=tf.as_dtype(t.dtype)
                    )
                else:
                    raise ValueError('Wrong dimension of target value. Can\'t be processed later.')
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
                    raise ValueError('Wrong depth of target value. Can\'t be processed later.')

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
    data = TFRecordDataset({
        'you': list(range(20)),
        'are': list(range(20)),
        'man': list(range(20)),
    })
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

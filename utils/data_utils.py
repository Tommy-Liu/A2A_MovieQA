import json
from functools import reduce
from operator import or_
from os.path import join

import numpy as np
import tensorflow as tf

from config import MovieQAConfig
from . import func_utils as fu

config = MovieQAConfig()
FILE_PATTERN = config.TFRECORD_PATTERN_
NPY_PATTERN_ = '%s.npy'


def pad_list_numpy(l, length):
    if isinstance(l[0], list):
        arr = np.zeros((len(l), length), dtype=np.int64)
        for idx, item in enumerate(l):
            arr[idx][:len(item)] = item
    else:
        arr = np.zeros(length, dtype=np.int64)
        arr[:len(l)] = l

    return arr


def json_dump(obj, file_name, ensure_ascii=False, indent=4):
    with open(file_name, 'w') as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)


def json_load(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def get_npy_name(feature_dir, video):
    return join(feature_dir, NPY_PATTERN_ % video)


def probe_type(value):
    return (vec_type_check(value) and reduce(or_, [probe_type(e) for e in value])) or \
           set([type(e) for e in value])


def type_check(value, dtype=(tuple, list, np.ndarray)):
    """Return true if value is one of default"""
    return isinstance(value, dtype)


def iter_type_check(value, dtype):
    return (vec_type_check(value) and all(iter_type_check(e, dtype) for e in value)) or \
           vec_type_check(value, dtype)


def vec_type_check(value, dtype=(tuple, list, np.ndarray)):
    return all(type_check(e, dtype) for e in value)


def matrix2d_type_check(value, dtype=(tuple, list, np.ndarray)):
    return all(vec_type_check(e, dtype) for e in value)


def to_feature(value, feature_list=False):
    """Wrapper of tensorflow feature"""

    def list_depth(l):
        return isinstance(l, list) and max(map(list_depth, l), default=0) + 1

    def zero_depth(_):
        return 0

    def num_dim(l):
        return l.ndim

    integer = int
    if type_check(value, np.ndarray):
        # numpy array
        numeric_check = np.issubsctype
        get_dim = num_dim
        integer = np.integer
    elif type_check(value, (tuple, list)):
        # list / tuple
        numeric_check = iter_type_check
        get_dim = list_depth
    else:
        # scalar
        numeric_check = isinstance
        get_dim = zero_depth
        integer = (int, np.integer)

    if numeric_check(value, integer):
        if get_dim(value) < 2:
            return int64_feature(value)
        elif get_dim(value) == 2:
            return int64_feature_list(value)
        else:
            raise ValueError('Too many dimensions (%d). At most 2.' % get_dim(value))
    elif numeric_check(value, float):
        if get_dim(value) < 2:
            return float_feature(value)
        elif get_dim(value) == 2:
            return float_feature_list(value)
        else:
            raise ValueError('Too many dimensions (%d). At most 2.' % get_dim(value))
    else:
        if get_dim(value) < 2:
            return bytes_feature(value)
        elif get_dim(value) == 2:
            return bytes_feature_list(value)
        else:
            raise ValueError('Too many dimensions (%d). At most 2.' % get_dim(value))


def int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    if not type_check(value):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    if not type_check(value):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    """Wrapper for inserting a float Feature into a SequenceExample proto."""
    if not type_check(value):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def float_feature_list(values):
    """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[float_feature(v) for v in values])


def qa_feature_example(example, subt, modality, is_training=False):
    example_feat = np.zeros((0, 1536), dtype=np.float32)
    example_subt = np.zeros((0, 41), dtype=np.int64)
    example_subt_length = []
    for name in example['feat']:
        f = np.load(name)
        s = subt[fu.get_base_name_without_ext(name)]

        assert len(f) == len(s['subtitle']), \
            "%s Video frames and subtitle are not aligned." % \
            fu.get_base_name_without_ext(name)

        if modality == 'fixed_num':
            num_sample = config.modality_config['fixed_num']
            if len(f) < num_sample:
                index = np.arange(len(f))
            else:
                index = np.linspace(0, len(f) - 1,
                                    num=num_sample,
                                    dtype=np.int64)

        elif modality == 'fixed_interval':
            num_interval = config.modality_config['fixed_interval']
            index = np.arange(0, len(f), step=num_interval, dtype=np.int64)

        elif modality == 'shot_major':
            num_interval = config.modality_config['shot_major']
            num_shot = np.amax(s['shot_boundary']) + 1
            index = np.array([], dtype=np.int64)
            for i in range(num_shot):
                arg = np.where(s['shot_boundary'] == i)[0]
                arg = arg[np.arange(0, arg.size, step=num_interval, dtype=np.int64)]
                index = np.concatenate([index, arg])

        elif modality == 'subtitle_major':
            num_interval = config.modality_config['subtitle_major']
            uniques = np.unique(s['subtitle_shot'])
            uniques = uniques[uniques > 0]
            index = np.array([], dtype=np.int64)
            for idx in uniques:
                arg = np.where(s['subtitle_shot'] == idx)[0]
                # print(np.arange(0, arg.size, step=num_interval, dtype=np.int64))
                arg = arg[np.arange(0, arg.size, step=num_interval, dtype=np.int64)]
                # print(arg)
                index = np.concatenate([index, arg])

        else:
            raise ValueError("Wrong modality.")

        example_feat = np.concatenate([example_feat, f[index]])
        example_subt = np.concatenate([example_subt, pad_list_numpy(s['subtitle'], 41)[index]])
        example_subt_length += [len(s['subtitle'][idx]) for idx in index]

    assert example_subt.size, "No subtitle!!"

    feature_lists = tf.train.FeatureLists(feature_list={
        "subt": to_feature(example_subt),
        "feat": to_feature(example_feat),
        "ans": to_feature(example['ans'])
    })
    feature = {
        "subt_length": to_feature(example_subt_length),
        "ans_length": to_feature(example['ans_length']),
        "ques": to_feature(example['ques']),
        "ques_length": to_feature(example['ques_length'])
    }

    if not is_training:
        feature['correct_index'] = to_feature(example['correct_index'])

    context = tf.train.Features(feature=feature)

    return tf.train.SequenceExample(context=context,
                                    feature_lists=feature_lists)


def qa_test_feature_parsed():
    context_features = {
        "subt_length": tf.VarLenFeature(dtype=tf.int64),
        "ans_length": tf.FixedLenFeature([5], dtype=tf.int64),
        "ques": tf.FixedLenFeature([25], dtype=tf.int64),
        "ques_length": tf.FixedLenFeature([], dtype=tf.int64),
        "correct_index": tf.FixedLenFeature([5], dtype=tf.int64)
    }
    sequence_features = {
        "subt": tf.FixedLenSequenceFeature([41], dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([1536], dtype=tf.float32),
        "ans": tf.FixedLenSequenceFeature([34], dtype=tf.int64)
    }
    return context_features, sequence_features


def qa_eval_feature_parsed():
    context_features = {
        "subt_length": tf.VarLenFeature(dtype=tf.int64),
        "ans_length": tf.FixedLenFeature([5], dtype=tf.int64),
        "ques": tf.FixedLenFeature([25], dtype=tf.int64),
        "ques_length": tf.FixedLenFeature([], dtype=tf.int64),
        "correct_index": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "subt": tf.FixedLenSequenceFeature([41], dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([1536], dtype=tf.float32),
        "ans": tf.FixedLenSequenceFeature([34], dtype=tf.int64)
    }
    return context_features, sequence_features


def qa_feature_parsed():
    context_features = {
        "subt_length": tf.VarLenFeature(dtype=tf.int64),
        "ans_length": tf.FixedLenFeature([2], dtype=tf.int64),
        "ques": tf.FixedLenFeature([25], dtype=tf.int64),
        "ques_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "subt": tf.FixedLenSequenceFeature([41], dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([1536], dtype=tf.float32),
        "ans": tf.FixedLenSequenceFeature([34], dtype=tf.int64)
    }
    return context_features, sequence_features


def get_dataset_name(d, name, split, modality, shard_id, num_shards, is_training=False):
    return join(d, FILE_PATTERN %
                (('training_' if is_training else ''),
                 name, split, modality, shard_id, num_shards))


def get_file_pattern(d, dataset_name, split, modality, num_shards, is_training):
    return join(d, MovieQAConfig.TFRECORD_FILE_PATTERN_ %
                (('training_' if is_training else ''),
                 dataset_name, split, modality, num_shards))

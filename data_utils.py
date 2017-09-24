import os
import re
import ujson as json
from os.path import join

import numpy as np
import tensorflow as tf

from config import MovieQAConfig

FILE_PATTERN = MovieQAConfig.TFRECORD_PATTERN_
NPY_PATTERN_ = '%s.npy'


class MovieQaDataLoader(object):
    def __init__(self):
        self.config = MovieQAConfig()
        self.qa = json.load(open(self.config.qa_file, 'r'))


def pad_list_numpy(l, length):
    if isinstance(l[0], list):
        arr = np.zeros((len(l), length), dtype=np.int64)
        for idx, item in enumerate(l):
            arr[idx][:len(item)] = item
    else:
        arr = np.zeros(length, dtype=np.int64)
        arr[:len(l)] = l

    return arr


def write_json(obj, file_name):
    with open(file_name, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def is_in(a, b):
    """
    Is a a subset of b ?
    """
    return set(a).issubset(set(b))


def get_npy_name(feature_dir, video):
    return join(feature_dir, NPY_PATTERN_ % video)


def exist_then_remove(f):
    if os.path.exists(f):
        os.remove(f)


def clean_token(l):
    """
    Clean up Subrip tags.
    """
    return re.sub(r'<.+?>', '', l)


def exist_make_dirs(d):
    """
    If the directory dose not exist, make one.
    """
    if not os.path.exists(d):
        os.makedirs(d)


# Wrapped function
def get_base_name_without_ext(p):
    """
    Get the base name without extension
    :param p: a string of directory or file path.
    :return: base name
    """
    base_name = get_base_name(p)
    base_name = os.path.splitext(base_name)[0]
    return base_name


# Fuck os.path.basename. I wrote my own version.
def get_base_name(p):
    """
    Get the subdirectory or file name
    in the last position of the path p.
    :param p: a string of directory or file path.
    :return: a string of base name.
    """
    pos = -1
    if p.split('/')[pos] == '':
        pos = -2
    return p.split('/')[pos]


def to_feature(value):
    """
    Wrapper of tensorflow feature
    :param value:
    :return:
    """
    if isinstance(value, np.ndarray):
        # value is ndarray
        if value.dtype == np.int64 or value.dtype == np.int32:
            # value is int
            if value.ndim > 1:
                # 2-d array
                return int64_feature_list(value)
            else:
                # 1-d array
                return int64_feature(value)
        elif value.dtype == np.float32 or value.dtype == np.float64:
            # value is float
            if value.shape[0] > 1:
                # 2-d array
                return float_feature_list(value)
            else:
                # 1-d array
                return float_feature(value)
    elif not isinstance(value, (tuple, list)):
        # value is scalar
        if isinstance(value, int):
            # int
            return int64_feature([value])
        elif isinstance(value, float):
            # float
            return float_feature([value])
        else:
            # string or byte
            return bytes_feature([str(value)])
    else:
        # value is list or tuple
        if isinstance(value[0], int):
            # int
            return int64_feature(value)
        elif isinstance(value[0], float):
            # float
            return float_feature(value)
        elif not isinstance(value[0], list):
            # string or byte
            return bytes_feature(value)
        else:
            # value is list of lists
            if isinstance(value[0][0], int):
                # int
                return int64_feature_list(value)
            elif isinstance(value[0][0], float):
                # float
                return float_feature_list(value)
            else:
                # string or byte
                return bytes_feature_list(value)


def int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature_list(values):
    return tf.train.FeatureList(feature=[float_feature(v) for v in values])


def qa_feature_example(example, subt, modality):
    example_feat = np.zeros((0, 1536), dtype=np.float32)
    example_subt = np.zeros((0, 41), dtype=np.int64)
    example_subt_length = []
    for name in example['feat']:
        f = np.load(name)
        s = subt[get_base_name_without_ext(name)]

        assert len(f) == len(s['subtitle']), \
            "%s Video frames and subtitle are not aligned." % \
            get_base_name_without_ext(name)

        if modality[0] == 'fixed_num':
            if len(f) < modality[1]:
                index = np.arange(len(f))
            else:
                index = np.linspace(0, len(f) - 1,
                                    num=modality[1],
                                    dtype=np.int64)

        elif modality[0] == 'fixed_interval':
            index = np.arange(0, len(f), step=modality[1])

        elif modality[0] == 'shot_major':
            num_shot = np.amax(s['shot_boundary']) + 1
            index = np.array([])
            for i in range(num_shot):
                arg = np.where(s['shot_boundary'] == i)
                if len(arg) < modality[1]:
                    index = np.concatenate([index, arg])
                else:
                    arg = np.choose(arg, np.linspace(0, len(arg) - 1,
                                                     num=modality[1]))
                    index = np.concatenate([index, arg])

        elif modality[0] == 'subtitle_major':
            uniques = np.unique(s['subtitle_index'])
            index = np.array([])
            for idx in uniques:
                arg = np.where(s['subtitle_index'] == idx)
                if len(arg) < modality[1]:
                    index = np.concatenate([index, arg])
                else:
                    arg = np.choose(arg, np.linspace(0, len(arg) - 1, num=modality[1]))
                    index = np.concatenate([index, arg])

        else:
            raise ValueError("Wrong modality.")

        example_feat = np.concatenate([example_feat, f[index]])
        example_subt = np.concatenate([example_subt, pad_list_numpy(s['subtitle'], 41)[index]])
        example_subt_length += [len(s['subtitle'][idx]) for idx in index]

    feature_lists = tf.train.FeatureLists(feature_list={
        "subt": to_feature(example_subt),
        "feat": to_feature(example_feat),
        "ans": to_feature(example['ans'])
    })
    context = tf.train.Features(feature={
        "subt_length": to_feature(example_subt_length),
        "ans_length": to_feature(example['ans_length']),
        "ques": to_feature(example['ques']),
        "ques_length": to_feature(example['ques_length'])
    })
    return tf.train.SequenceExample(context=context,
                                    feature_lists=feature_lists)


def qa_eval_feature_example(example, subt, split, modality):
    example_feat = np.zeros((0, 1536), dtype=np.float32)
    example_subt = np.zeros((0, 41), dtype=np.int64)
    example_subt_length = []
    for name in example['feat']:
        f = np.load(name)
        s = subt[get_base_name_without_ext(name)]

        assert len(f) == len(s['subtitle']), \
            "%s Video frames and subtitle are not aligned." % \
            get_base_name_without_ext(name)

        if modality[0] == 'fixed_num':
            if len(f) < modality[1]:
                index = np.arange(len(f))
            else:
                index = np.linspace(0, len(f) - 1,
                                    num=modality[1],
                                    dtype=np.int64)

        elif modality[0] == 'fixed_interval':
            index = np.arange(0, len(f), step=modality[1])

        elif modality[0] == 'shot_major':
            num_shot = np.amax(s['shot_boundary']) + 1
            index = np.array([])
            for i in range(num_shot):
                arg = np.where(s['shot_boundary'] == i)
                if len(arg) < modality[1]:
                    index = np.concatenate([index, arg])
                else:
                    arg = np.choose(arg, np.linspace(0, len(arg) - 1,
                                                     num=modality[1]))
                    index = np.concatenate([index, arg])

        elif modality[0] == 'subtitle_major':
            uniques = np.unique(s['subtitle_index'])
            index = np.array([])
            for idx in uniques:
                arg = np.where(s['subtitle_index'] == idx)
                if len(arg) < modality[1]:
                    index = np.concatenate([index, arg])
                else:
                    arg = np.choose(arg, np.linspace(0, len(arg) - 1, num=modality[1]))
                    index = np.concatenate([index, arg])

        else:
            raise ValueError("Wrong modality.")

        example_feat = np.concatenate([example_feat, f[index]])
        example_subt = np.concatenate([example_subt, pad_list_numpy(s['subtitle'], 41)[index]])
        example_subt_length += [len(s['subtitle'][idx]) for idx in index]

    feature_lists = tf.train.FeatureLists(feature_list={
        "subt": to_feature(example_subt),
        "feat": to_feature(example_feat),
        "ans": to_feature(example['ans'])
    })
    feature = {
        "subt_length": to_feature(example_subt_length),
        "ans_length": to_feature(example['ans_length']),
        "ques": to_feature(example['ques']),
        "ques_length": to_feature(example['ques_length']),
        "correct_index": to_feature(example['correct_index'])
    }
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
import os
import re
from os.path import join

import numpy as np
import tensorflow as tf

FILE_PATTERN = '%s_%s_%05d-of-%05d.tfrecord'
NPY_PATTERN_ = '%s.npy'


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
    return re.sub(r'<.*?>', '', l)


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
    if isinstance(value, np.ndarray):
        # value is ndarray
        if value.dtype == np.int64 or value.dtype == np.int32:
            # value is int
            if value.shape[0] > 1:
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


def qa_eval_feature_example(example, split):
    feat = np.concatenate([np.load(name) for name in example['feat']],
                          axis=0).astype(np.float32)

    feature_lists = tf.train.FeatureLists(feature_list={
        "subt": to_feature(example['subt']),
        "feat": to_feature(feat),
        "ans": to_feature(example['ans'])
    })
    feature = {
        "subt_length": to_feature(example['subt_length']),
        "ans_length": to_feature(example['ans_length']),
        "ques": to_feature(example['ques']),
        "ques_length": to_feature(example['ques_length'])
    }
    if split != 'test':
        feature['correct_index'] = to_feature(example['correct_index'])
    context = tf.train.Features(feature=feature)
    return tf.train.SequenceExample(context=context,
                                    feature_lists=feature_lists)


def qa_test_feature_parsed():
    context_features = {
        "subt_length": tf.VarLenFeature(dtype=tf.int64),
        "ans_length": tf.VarLenFeature(dtype=tf.int64),
        "ques": tf.VarLenFeature(dtype=tf.int64),
        "ques_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "subt": tf.VarLenFeature(dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([1536], dtype=tf.float32),
        "ans": tf.VarLenFeature(dtype=tf.int64)
    }
    return context_features, sequence_features

def qa_eval_feature_parsed():
    context_features = {
        "subt_length": tf.VarLenFeature(dtype=tf.int64),
        "ans_length": tf.VarLenFeature(dtype=tf.int64),
        "ques": tf.VarLenFeature(dtype=tf.int64),
        "ques_length": tf.FixedLenFeature([], dtype=tf.int64),
        "correct_index": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "subt": tf.VarLenFeature(dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([1536], dtype=tf.float32),
        "ans": tf.VarLenFeature(dtype=tf.int64)
    }
    return context_features, sequence_features


def qa_feature_parsed():
    context_features = {
        "subt_length": tf.VarLenFeature(dtype=tf.int64),
        "ans_length": tf.VarLenFeature(dtype=tf.int64),
        "ques": tf.VarLenFeature(dtype=tf.int64),
        "ques_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "subt": tf.VarLenFeature(dtype=tf.int64),
        "feat": tf.FixedLenSequenceFeature([1536], dtype=tf.float32),
        "ans": tf.VarLenFeature(dtype=tf.int64)
    }
    return context_features, sequence_features


def qa_feature_example(example):
    feat = np.concatenate([np.load(name) for name in example['feat']],
                          axis=0).astype(np.float32)
    assert len(example['subt']) == len(feat), \
        "Fuck my life... %s" % ' '.join(example['video_clips'])
    feature_lists = tf.train.FeatureLists(feature_list={
        "subt": to_feature(example['subt']),
        "feat": to_feature(feat),
        "ans": to_feature(example['ans'])
    })
    context = tf.train.Features(feature={
        "subt_length": to_feature(example['subt_length']),
        "ans_length": to_feature(example['ans_length']),
        "ques": to_feature(example['ques']),
        "ques_length": to_feature(example['ques_length'])
    })
    return tf.train.SequenceExample(context=context,
                                    feature_lists=feature_lists)


def get_dataset_name(d, name, split, shard_id, num_shards, is_training=False):
    if is_training:
        return join(d, 'training_' + FILE_PATTERN % (name, split, shard_id, num_shards))
    else:
        return join(d, FILE_PATTERN % (name, split, shard_id, num_shards))


def frame_feature_example(features):
    frame_feats = float_feature_list([feats.tolist() for feats in features])
    context = tf.train.Features(feature={
        "number": int64_feature(len(features))
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        "frame_feats": frame_feats
    })
    return tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

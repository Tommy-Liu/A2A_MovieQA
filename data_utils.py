import os
import re
import ujson as json
from os.path import join, exists

import time
import io
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from config import MovieQAConfig

config = MovieQAConfig()
FILE_PATTERN = config.TFRECORD_PATTERN_
NPY_PATTERN_ = '%s.npy'


def load_embedding_vec(target, embedding_size=300):
    start_time = time.time()
    if target == 'glove':
        key_file = config.glove_embedding_key_file
        vec_file = config.glove_embedding_vec_file
        raw_file = config.glove_file
        load_fn = load_glove
    elif target == 'w2v':
        key_file = config.w2v_embedding_key_file
        vec_file = config.w2v_embedding_vec_file
        raw_file = config.word2vec_file
        load_fn = load_w2v
    elif target == 'fasttext':
        key_file = config.ft_embedding_key_file
        vec_file = config.ft_embedding_vec_file
        raw_file = config.fasttext_file
        load_fn = load_glove
    else:
        key_file = None
        vec_file = None
        raw_file = None
        load_fn = None

    if exists(key_file) and exists(vec_file):
        embedding_keys = jload(key_file)
        embedding_vecs = np.load(vec_file)
    else:
        embedding = load_fn(raw_file)
        embedding_keys = []
        embedding_vecs = np.zeros((len(embedding), embedding_size), dtype=np.float32)
        for i, k in enumerate(embedding.keys()):
            embedding_keys.append(k)
            embedding_vecs[i] = embedding[k]
        jdump(embedding_keys, key_file)
        np.save(vec_file, embedding_vecs)

    print('Loading embedding done. %.3f s' % (time.time() - start_time))
    return embedding_keys, embedding_vecs

def load_w2v(file):
    embedding = {}

    with open(file, 'r') as f:
        num, dim = [int(comp) for comp in f.readline().strip().split()]
        for _ in trange(num, desc='Load word embedding %dd' % dim):
            word, *vec = f.readline().rstrip().rsplit(sep=' ', maxsplit=dim)
            vec = [float(e) for e in vec]
            embedding[word] = vec
        assert len(embedding) == num, 'Wrong size of embedding.'
    return embedding

def load_glove(filename, embedding_size=300):
    embedding = {}

    # Read in the data.
    with io.open(filename, 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(tqdm(savefile)):
            tokens = line.rstrip().split(sep=' ', maxsplit=embedding_size)

            word, *entries = tokens

            embedding[word] = [float(x) for x in entries]
            assert len(embedding[word]) == embedding_size, 'Wrong embedding dim.'

    return embedding

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


def jdump(obj, file_name, ensure_ascii=False, indent=4):
    with open(file_name, 'w') as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)


def jload(file_name):
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


def get_imdb_key(base_name):
    return base_name.split('.')[0]


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


def iter_type_check(value, dtype):
    return all(isinstance(e, dtype) for e in value)


def recur_iter_type_check(value, dtype):
    return all(iter_type_check(e, dtype) for e in value)


def to_feature(value):
    """
    Wrapper of tensorflow feature
    :param value:
    :return:
    """
    if isinstance(value, np.ndarray):
        # value is ndarray
        if np.issubdtype(value.dtype, np.integer):
            # value is int
            if value.ndim == 2:
                # 2-d array
                return int64_feature_list(value)
            elif value.ndim == 1:
                # 1-d array
                return int64_feature(value)
            else:
                raise ValueError('Too many dimensions.')
        elif np.issubdtype(value.dtype, np.floating):
            # value is float
            if value.ndim == 2:
                # 2-d array
                return float_feature_list(value)
            elif value.ndim == 1:
                # 1-d array
                return float_feature(value)
            else:
                raise ValueError('Too many dimensions.')
    elif isinstance(value, (tuple, list)):
        # value is list or tuple
        if iter_type_check(value, int):
            # int
            return int64_feature(value)
        elif iter_type_check(value, float):
            # float
            return float_feature(value)
        elif iter_type_check(value, list):
            # value is list of lists
            if recur_iter_type_check(value, int):
                # int
                return int64_feature_list(value)
            elif recur_iter_type_check(value, float):
                # float
                return float_feature_list(value)
            else:
                # string or byte
                return bytes_feature_list(value)
        else:
            # string or byte
            return bytes_feature(value)

    else:
        # value is scalar
        if isinstance(value, (int, np.integer)):
            # int
            return int64_feature([value])
        elif isinstance(value, (float, np.floating)):
            # float
            return float_feature([value])
        else:
            # string or byte
            return bytes_feature([str(value)])


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


def qa_feature_example(example, subt, modality, is_training=False):
    example_feat = np.zeros((0, 1536), dtype=np.float32)
    example_subt = np.zeros((0, 41), dtype=np.int64)
    example_subt_length = []
    for name in example['feat']:
        f = np.load(name)
        s = subt[get_base_name_without_ext(name)]

        assert len(f) == len(s['subtitle']), \
            "%s Video frames and subtitle are not aligned." % \
            get_base_name_without_ext(name)

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


def pprint(s, ch='='):
    print(ch * (max([len(e) for e in s]) + 5))
    print('\n'.join(s))
    print(ch * (max([len(e) for e in s]) + 5))

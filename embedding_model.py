import argparse
import math
import time
from collections import Counter
from glob import glob
from multiprocessing import Pool
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset
from tqdm import tqdm, trange

import data_utils as du
from config import MovieQAConfig
from qa_preprocessing import load_embedding

UNK = 'UNK'
RECORD_FILE_PATTERN = join('./data', 'dataset', 'embedding_%05d-of-%05d.tfrecord')

config = MovieQAConfig()


def parser(record):
    features = {
        "vec": tf.FixedLenFeature([300], tf.float32),
        "word": tf.FixedLenFeature([98], tf.int64),
        "length": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, features)

    return parsed['vec'], parsed['word'], parsed['length']


class EmbeddingData(object):
    RECORD_FILE_PATTERN_ = join('./data', 'dataset', 'embedding_*.tfrecord')

    def __init__(self):
        self.file_names = glob(self.RECORD_FILE_PATTERN_)
        self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
        dataset = TFRecordDataset(self.file_names_placeholder)
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.batch(64)
        self.iterator = self.dataset.make_initializable_iterator()

    def build_char_vocab(self):
        pass


def create_one_example(v, w, l):
    feature = {
        "vec": du.to_feature(np.squeeze(v)),
        "word": du.to_feature(np.squeeze(w)),
        "len": du.to_feature(int(l)),
    }

    features = tf.train.Features(feature=feature)

    return tf.train.Example(features=features)


def create_one_record(ex_tuple):
    shard_id, example_list = ex_tuple
    output_filename = RECORD_FILE_PATTERN % (shard_id + 1, args.num_shards)
    du.exist_then_remove(output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(example_list)):
            embedding_vec, embedding_word, embedding_word_length = example_list[i]
            example = create_one_example(embedding_vec, embedding_word, embedding_word_length)
            tfrecord_writer.write(example.SerializeToString())


def create_records():
    start_time = time.time()
    embedding_vec = np.load(config.w2v_embedding_npy_file)
    embedding_word = np.load(config.encode_embedding_file)
    embedding_word_length = np.load(config.encode_embedding_len_file)
    print('Loading file done. Spend %f sec' % (time.time() - start_time))
    num_per_shard = int(math.ceil(len(embedding_word_length) / float(args.num_shards)))
    example_list = []
    for j in trange(args.num_shards):
        start_ndx = j * num_per_shard
        end_ndx = min((j + 1) * num_per_shard, len(embedding_word_length))
        example_list.append((j, [(embedding_vec[i, :], embedding_word[i, :], embedding_word_length[i])
                                 for i in range(start_ndx, end_ndx)]))
    with Pool(8) as pool, tqdm(total=args.num_shards, desc='Tfrecord') as pbar:
        for _ in pool.imap_unordered(create_one_record, example_list):
            pbar.update()


def process():
    tokenize_qa = du.load_json(config.avail_tokenize_qa_file)
    subtitle = du.load_json(config.subtitle_file)
    embedding = load_embedding(config.word2vec_file)

    embed_char_counter = Counter()
    for k in tqdm(embedding.keys()):
        embed_char_counter.update(k)

    embedding_keys = list(embedding.keys())
    embedding_array = np.array(list(embedding.values()), dtype=np.float32)
    du.write_json(embedding_keys, config.w2v_embedding_file)
    np.save(config.w2v_embedding_npy_file,
            embedding_array)

    qa_char_counter = Counter()
    for k in tokenize_qa.keys():
        for qa in tqdm(tokenize_qa[k], desc='Char counting %s' % k):
            for w in qa['tokenize_question']:
                qa_char_counter.update(w)
            for a in qa['tokenize_answer']:
                for w in a:
                    qa_char_counter.update(w)
            for v in qa['video_clips']:
                for l in subtitle[v]:
                    for w in l:
                        qa_char_counter.update(w)

    du.write_json(embed_char_counter, config.embed_char_counter_file)
    du.write_json(qa_char_counter, config.qa_char_counter_file)

    count_array = np.array(list(embed_char_counter.values()), dtype=np.float32)
    m, v, md, f = np.mean(count_array), np.std(count_array), np.median(count_array), np.percentile(count_array, 95)
    print(m, v, md, f)

    above_mean = dict(filter(lambda item: item[1] > f, embed_char_counter.items()))
    below_mean = dict(filter(lambda item: item[1] < f, embed_char_counter.items()))
    below_occur = set(filter(lambda k: k in qa_char_counter, below_mean.keys()))
    final_set = below_occur.union(set(above_mean.keys()))
    du.write_json(list(final_set) + [UNK], config.char_vocab_file)

    vocab = du.load_json(config.char_vocab_file)
    encode_embedding_keys = np.zeros((len(embedding_keys), 98), dtype=np.int64)
    length = np.zeros(len(embedding_keys), dtype=np.int64)
    for i, k in enumerate(tqdm(embedding_keys, desc='OP')):
        encode_embedding_keys[i, :len(k)] = [
            vocab.index(ch) if ch in vocab else vocab.index('UNK')
            for ch in k
        ]
        length[i] = len(k)

    np.save(config.encode_embedding_file, encode_embedding_keys)
    np.save(config.encode_embedding_len_file, length)


def main():
    data = EmbeddingData()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true',
                        help='Process the data which creating tfrecords needs.')
    parser.add_argument('--num_shards', default=128, help='Number of tfrecords.')
    parser.add_argument('--tfrecord', action='store_true', help='Create tfrecords.')
    args = parser.parse_args()
    if args.process:
        process()
    if args.tfrecord:
        create_records()

        # main()

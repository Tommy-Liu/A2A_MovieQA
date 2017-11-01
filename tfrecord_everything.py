# import argparse
import math
from functools import partial
from multiprocessing import Pool

import tensorflow as tf
from tqdm import trange, tqdm

import data_utils as du


def resolve_feature():


def create_one_tfrecord(RECORD_FILE_PATTERN, keys, num_shards, ex_tuple):
    shard_id, example_list = ex_tuple
    output_filename = RECORD_FILE_PATTERN % (shard_id + 1, num_shards)
    du.exist_then_remove(output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(example_list)):
            embedding_vec, embedding_word, embedding_word_length = example_list[i]
            example = create_one_example(embedding_vec, embedding_word, embedding_word_length)
            tfrecord_writer.write(example.SerializeToString())

class TfrecordDataset(object):
    TFRECORD_PATTERN = '_%05d-of-%05d.tfrecord'
    TFRECORD_PATTERN_ = '*.tfrecord'

    def __init__(self, target, dset_name='default', dset_dir='./data/dataset', num_shards=128, num_threads=8):
        self.num_example, *check_tail = set(len(v) for v in target.values())
        assert len(check_tail) > 0, 'Different length of targets.'
        num_per_shard = int(math.ceil(self.num_example / float(num_shards)))
        example_list = []
        for j in trange(num_shards):
            start_ndx = j * num_per_shard
            end_ndx = min((j + 1) * num_per_shard, self.num_example)
            example_list.append((j, [(v[i] for v in target.values())
                                     for i in range(start_ndx, end_ndx)]))
        func = partial(create_one_tfrecord, self.TFRECORD_PATTERN, list(target.keys()), num_shards)
        with Pool(num_threads) as pool, tqdm(total=num_shards, desc=dset_name) as pbar:
            for _ in pool.


if __name__ == '__main__':
    data = TfrecordDataset({
        'you': list(range(20)),
        'are': list(range(20)),
        'man': list(range(20)),
    })
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

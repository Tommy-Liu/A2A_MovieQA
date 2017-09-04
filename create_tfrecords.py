import tensorflow as tf
import numpy as np
import random
import json
import math

from data_utils import *
from video_preprocessing import get_base_name_without_ext
from extract_feature import get_npy_name

flags = tf.app.flags
flags.DEFINE_integer("num_shards", 5, "")
flags.DEFINE_string("dataset_name", "movieqa", "")
flags.DEFINE_string("dataset_dir", "./dataset", "")
FLAGS = flags.FLAGS


# 1: dataset name, 2:split name, 3: shard id, 4: total shard number

# ['avail_qa_train', 'avail_qa_test', 'avail_qa_val']
# ['qid', 'question', 'answers', 'imdb_key', 'correct_index', 'plot_alignment',
# 'video_clips', 'tokenize_question', 'tokenize_answer', 'tokenize_video_subtitle',
# 'encoded_answer', 'encoded_question', 'encoded_subtitle']

def create_tfrecord(qas, split):
    num_per_shard = 5  # int(math.ceil(len(qas) / float(FLAGS.num_shards)))
    for shard_id in range(FLAGS.num_shard):
        output_filename = get_dataset_name(FLAGS.dataset_dir,
                                           FLAGS.dataset_name,
                                           split,
                                           shard_id + 1,
                                           FLAGS.num_shards)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(qas))
            for i in range(start_ndx, end_ndx):
                for ans_idx in range(len(qas[i]['encoded_answer'])):
                    if ans_idx == qas[i]['correct_index']:
                        for _ in range(len(qas[i]['encoded_answer']) - 1):
                            example = qa_feature_example(qas[i], ans_idx)
                            tfrecord_writer.write(example.SerializeToString())
                    else:
                        example = qa_feature_example(qas[i], ans_idx)
                        tfrecord_writer.write(example.SerializeToString())

def main():
    avail_preprocessing_qa = json.load(open('./avail_preprocessing_qa.json', 'r'))
    create_tfrecord(avail_preprocessing_qa['avail_qa_train'], split='train')
    create_tfrecord(avail_preprocessing_qa['avail_qa_test'], split='test')
    create_tfrecord(avail_preprocessing_qa['avail_qa_val'], split='val')


if __name__ == '__main__':
    tf.app.run()

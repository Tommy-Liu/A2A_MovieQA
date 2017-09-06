import tensorflow as tf
import numpy as np
import random
import json
import math

from tqdm import trange
from data_utils import get_dataset_name, qa_feature_example
from video_preprocessing import exist_make_dirs

flags = tf.app.flags
flags.DEFINE_integer("num_shards", 5, "")
flags.DEFINE_string("dataset_name", "movieqa", "")
flags.DEFINE_string("dataset_dir", "./dataset", "")
flags.DEFINE_string("encode_file_name", "./encode_qa.json",
                    "")
FLAGS = flags.FLAGS


# 1: dataset name, 2:split name, 3: shard id, 4: total shard number

# ['avail_qa_train', 'avail_qa_test', 'avail_qa_val']
# ['qid', 'question', 'answers', 'imdb_key', 'correct_index', 'plot_alignment',
# 'video_clips', 'tokenize_question', 'tokenize_answer', 'tokenize_video_subtitle',
# 'encoded_answer', 'encoded_question', 'encoded_subtitle']

def create_tfrecord(qas, split):
    num_per_shard = 5  # int(math.ceil(len(qas) / float(FLAGS.num_shards)))
    for shard_id in trange(FLAGS.num_shards,
                           desc="shard loop"):
        output_filename = get_dataset_name(FLAGS.dataset_dir,
                                           FLAGS.dataset_name,
                                           split,
                                           shard_id + 1,
                                           FLAGS.num_shards)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(qas))
            for i in trange(start_ndx, end_ndx,
                            desc="shard %d" % (shard_id + 1)):
                for ans_idx in trange(len(qas[i]['encoded_answer']),
                                      desc="answer loop"):
                    if ans_idx == qas[i]['correct_index']:
                        for _ in trange(len(qas[i]['encoded_answer']) - 1,
                                        desc="duplicate loop"):
                            example = qa_feature_example(qas[i], ans_idx)
                            tfrecord_writer.write(example.SerializeToString())
                    # TODO(tommy8054): Decide training process
                    elif True:
                        example = qa_feature_example(qas[i], ans_idx)
                        tfrecord_writer.write(example.SerializeToString())


def main(_):
    encode_qa = json.load(open(FLAGS.encode_file_name, 'r'))
    exist_make_dirs(FLAGS.dataset_dir)
    create_tfrecord(encode_qa['encode_qa_train'], split='train')
    create_tfrecord(encode_qa['encode_qa_test'], split='test')
    create_tfrecord(encode_qa['encode_qa_val'], split='val')


if __name__ == '__main__':
    tf.app.run()

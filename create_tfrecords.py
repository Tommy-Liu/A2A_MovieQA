import json
import math
from functools import partial
from multiprocessing import Pool

import tensorflow as tf
from tqdm import tqdm

from config import MovieQAConfig
from data_utils import get_dataset_name, qa_feature_example, \
    qa_eval_feature_example, exist_make_dirs, exist_then_remove

config = MovieQAConfig()


# 1: dataset name, 2:split name, 3: shard id, 4: total shard number

# ['avail_qa_train', 'avail_qa_test', 'avail_qa_val']
# ['qid', 'question', 'answers', 'imdb_key', 'correct_index', 'plot_alignment',
# 'video_clips', 'tokenize_question', 'tokenize_answer', 'tokenize_video_subtitle',
# 'encoded_answer', 'encoded_question', 'encoded_subtitle']

def create_one_tfrecord(split, num_per_shard, qas, is_training, shard_id):
    output_filename = get_dataset_name(config.dataset_dir,
                                       config.dataset_name,
                                       split,
                                       shard_id + 1,
                                       config.num_shards,
                                       is_training)
    exist_then_remove(output_filename)
    # print('Start writing %s.' % output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(qas))
        for i in range(start_ndx, end_ndx):
            # trange(start_ndx, end_ndx,  #
            #         desc="shard %d" % (shard_id + 1)):
            if is_training:
                for ans_idx in range(len(qas[i]['encoded_answer'])):
                    # trange(len(qas[i]['encoded_answer']),  #
                    #               desc="answer loop"):
                    if ans_idx != qas[i]['correct_index'] and qas[i]['encoded_answer'][ans_idx] != []:
                        # print(qas[i]['encoded_answer'][ans_idx])
                        example = qa_feature_example(qas[i], config.feature_dir, ans_idx)
                        tfrecord_writer.write(example.SerializeToString())
            else:
                example = qa_eval_feature_example(qas[i], config.feature_dir)
                tfrecord_writer.write(example.SerializeToString())
                # print('Writing %s done!' % output_filename)


def create_tfrecord(qas, split, is_training=False):
    num_per_shard = int(math.ceil(len(qas) / float(config.num_shards)))
    shard_id_list = list(range(config.num_shards))
    func = partial(create_one_tfrecord,
                   split,
                   num_per_shard,
                   qas,
                   is_training)
    with Pool(8) as p, tqdm(total=config.num_shards, desc="Write tfrecords") as pbar:
        for i, _ in enumerate(p.imap_unordered(func, shard_id_list)):
            pbar.update()
            # for shard_id in trange(config.num_shards,  # range(config.num_shards):
            #                        desc="shard loop"):
            #
            #     output_filename = get_dataset_name(config.dataset_dir,
            #                                        config.dataset_name,
            #                                        split,
            #                                        shard_id + 1,
            #                                        config.num_shards,
            #                                        is_training)
            #     exist_then_remove(output_filename)
            #     with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            #         start_ndx = shard_id * num_per_shard
            #         end_ndx = min((shard_id + 1) * num_per_shard, len(qas))
            #         for i in trange(start_ndx, end_ndx,  # range(start_ndx, end_ndx):
            #                         desc="shard %d" % (shard_id + 1)):
            #             if is_training:
            #                 for ans_idx in trange(len(qas[i]['encoded_answer']),  # range(len(qas[i]['encoded_answer'])):
            #                                       desc="answer loop"):
            #                     if ans_idx != qas[i]['correct_index'] and qas[i]['encoded_answer'][ans_idx] != []:
            #                         # print(qas[i]['encoded_answer'][ans_idx])
            #                         example = qa_feature_example(qas[i], config.feature_dir, ans_idx)
            #                         tfrecord_writer.write(example.SerializeToString())
            #             else:
            #                 example = qa_eval_feature_example(qas[i], config.feature_dir)
            #                 tfrecord_writer.write(example.SerializeToString())


def main(_):
    encode_qa = json.load(open(config.encode_file_name, 'r'))
    print('Json file loading done !!')
    exist_make_dirs(config.dataset_dir)
    create_tfrecord(encode_qa['encode_qa_train'], split='train', is_training=True)
    # create_tfrecord(encode_qa['encode_qa_train'], split='train')
    # create_tfrecord(encode_qa['encode_qa_test'], split='test')
    # create_tfrecord(encode_qa['encode_qa_val'], split='val')


if __name__ == '__main__':
    tf.app.run()

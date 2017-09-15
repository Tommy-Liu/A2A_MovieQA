import json
import math
from functools import partial
from multiprocessing import Pool

import tensorflow as tf
from tqdm import tqdm

from config import MovieQAConfig
from data_utils import get_dataset_name, qa_feature_example, \
    qa_eval_feature_example, exist_make_dirs, exist_then_remove, \
    get_npy_name, get_base_name_without_ext

config = MovieQAConfig()


# 1: dataset name, 2:split name, 3: shard id, 4: total shard number

# ['avail_qa_train', 'avail_qa_test', 'avail_qa_val']
# ['qid', 'question', 'answers', 'imdb_key', 'correct_index', 'plot_alignment',
# 'video_clips', 'tokenize_question', 'tokenize_answer', 'tokenize_video_subtitle',
# 'encoded_answer', 'encoded_question', 'encoded_subtitle']

def create_one_tfrecord(split, num_per_shard, example_list, is_training, shard_id):
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
        end_ndx = min((shard_id + 1) * num_per_shard, len(example_list))
        for i in range(start_ndx, end_ndx):
            # trange(start_ndx, end_ndx,  #
            #         desc="shard %d" % (shard_id + 1)):
            if is_training:
                example = qa_feature_example(example_list[i])
                tfrecord_writer.write(example.SerializeToString())
            else:
                example = qa_eval_feature_example(example_list[i], split)
                tfrecord_writer.write(example.SerializeToString())
                # print('Writing %s done!' % output_filename)


def get_total_example(qas, split, is_training=False):
    example_list = []
    if is_training:
        for qa in tqdm(qas, desc="Get total examples"):
            for ans_idx in range(len(qa['encoded_answer'])):
                if ans_idx != qa['correct_index'] and qa['encoded_answer'][ans_idx] != []:
                    example = {
                        "subt": qa['encoded_subtitle'],
                        "feat": [get_npy_name(config.feature_dir, get_base_name_without_ext(v))
                                 for v in qa['video_clips']],
                        "ques": qa['encoded_question'],
                        "ans": [qa['encoded_answer'][qa['correct_index']],
                                qa['encoded_answer'][ans_idx]],
                        "subt_length": [len(sent) for sent in qa['encoded_subtitle']],
                        "ques_length": len(qa['encoded_question']),
                        "ans_length": [len(qa['encoded_answer'][qa['correct_index']]),
                                       len(qa['encoded_answer'][ans_idx])],
                        "video_clips": qa['video_clips']
                    }
                    example_list.append(example)
    else:
        for qa in tqdm(qas, desc="Get total examples"):
            example = {
                "subt": qa['encoded_subtitle'],
                "feat": [get_npy_name(config.feature_dir, get_base_name_without_ext(v))
                         for v in qa['video_clips']],
                "ques": qa['encoded_question'],
                "subt_length": [len(sent) for sent in qa['encoded_subtitle']],
                "ques_length": len(qa['encoded_question']),
                "video_clips": qa['video_clips']
            }
            if split != 'test':
                example['correct_index'] = qa['correct_index']
            ans = []
            ans_length = []
            pad_a = []
            for a in qa['encoded_answer']:
                if a:
                    pad_a = a
                    break
            for a in qa['encoded_answer']:
                if not a:
                    ans.append(pad_a)
                    ans_length.append(len(pad_a))
                else:
                    ans.append(a)
                    ans_length.append(len(a))

            example['ans'] = ans
            example['ans_length'] = ans_length
            example_list.append(example)
    return example_list


def create_tfrecord(qas, split, is_training=False):
    example_list = get_total_example(qas, split, is_training)
    config.update_info({
        "num_%s%s_examples" %
        ("training_" if is_training else "",
         split): len(example_list)
    })
    num_per_shard = int(math.ceil(len(example_list) / float(config.num_shards)))
    shard_id_list = list(range(config.num_shards))
    func = partial(create_one_tfrecord,
                   split,
                   num_per_shard,
                   example_list,
                   is_training)
    with Pool(8) as p, tqdm(total=config.num_shards, desc="Write tfrecords") as pbar:
        for i, _ in enumerate(p.imap_unordered(func, shard_id_list)):
            pbar.update()


def main(_):
    encode_qa = json.load(open(config.avail_encode_qa_file, 'r'))
    print('Json file loading done !!')
    exist_make_dirs(config.dataset_dir)
    create_tfrecord(encode_qa['encode_qa_%s' % FLAGS.split],
                    split=FLAGS.split,
                    is_training=FLAGS.is_training)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("split", "train", "")
    flags.DEFINE_bool("is_training", False, "")
    FLAGS = flags.FLAGS
    tf.app.run()

import math
import ujson as json
from functools import partial
from glob import glob
from multiprocessing import Pool, Manager
from os.path import join

import tensorflow as tf
from tqdm import tqdm

import data_utils as du
from config import MovieQAConfig

config = MovieQAConfig()


# 1: dataset name, 2:split name, 3: shard id, 4: total shard number

# ['avail_qa_train', 'avail_qa_test', 'avail_qa_val']
# ['qid', 'question', 'answers', 'imdb_key', 'correct_index', 'plot_alignment',
# 'video_clips', 'tokenize_question', 'tokenize_answer', 'tokenize_video_subtitle',
# 'encoded_answer', 'encoded_question', 'encoded_subtitle']


def create_one_tfrecord(split, modality, num_per_shard, example_list, subt, is_training, shard_id):
    output_filename = du.get_dataset_name(config.dataset_dir,
                                          config.dataset_name,
                                          split,
                                          modality,
                                          shard_id + 1,
                                          config.num_shards,
                                          is_training)
    du.exist_then_remove(output_filename)
    # print('Start writing %s.' % output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(example_list))
        for i in range(start_ndx, end_ndx):
            # trange(start_ndx, end_ndx,  #
            #         desc="shard %d" % (shard_id + 1)):
            if is_training:
                example = du.qa_feature_example(example_list[i], subt, (modality, config.modality[modality]))
                tfrecord_writer.write(example.SerializeToString())
            else:
                example = du.qa_eval_feature_example(example_list[i], subt, split,
                                                     (modality, config.modality[modality]))
                tfrecord_writer.write(example.SerializeToString())
                # print('Writing %s done!' % output_filename)


def get_total_example(qas, split, is_training=False):
    example_list = []
    if is_training:
        for qa in tqdm(qas, desc="Get total examples"):
            for ans_idx in range(len(qa['encoded_answer'])):
                if ans_idx != qa['correct_index'] and qa['encoded_answer'][ans_idx]:
                    example_list.append({
                        "feat": [du.get_npy_name(config.feature_dir, v)
                                 for v in qa['video_clips']],
                        "ques": du.pad_list_numpy(qa['encoded_question'], config.ques_max_length),
                        "ans": du.pad_list_numpy([qa['encoded_answer'][qa['correct_index']],
                                                  qa['encoded_answer'][ans_idx]], config.ans_max_length),
                        "ques_length": len(qa['encoded_question']),
                        "ans_length": [len(qa['encoded_answer'][qa['correct_index']]),
                                       len(qa['encoded_answer'][ans_idx])],
                        "video_clips": qa['video_clips']
                    })
    else:
        for qa in tqdm(qas, desc="Get total examples"):
            if split != 'test':
                ans = [a for a in qa['encoded_answer']
                       if a and a != qa['encoded_answer'][qa['correct_index']]]
                assert all(ans), "Empty answer occurs!\n %s" % json.dumps(qa, indent=4)
                ans.append(qa['encoded_answer'][qa['correct_index']])
                correct_index = len(ans) - 1
                for _ in range(5 - len(ans)):
                    ans.append(ans[0])
            else:
                ans = []
                correct_index = []
                for idx, a in enumerate(qa['encoded_answer']):
                    if a:
                        ans.append(a)
                        correct_index.append(idx)

                for _ in range(5 - len(ans)):
                    ans.append(ans[0])
                    correct_index.append(correct_index[0])
                assert all(ans), "Empty answer occurs!\n %s" % json.dumps(qa, indent=4)
            ans_length = [len(a) for a in ans]

            example_list.append({
                "feat": [du.get_npy_name(config.feature_dir, v)
                         for v in qa['video_clips']],
                "ques": qa['encoded_question'],
                "ques_length": len(qa['encoded_question']),
                "video_clips": qa['video_clips'],
                "ans": ans,
                "ans_length": ans_length,
                "correct_index": correct_index
            })
    return example_list


def create_tfrecord(qas, subt, split, modality, is_training=False):
    example_list = get_total_example(qas, split, is_training)
    config.update_info({
        "num_%s%s_examples" %
        ("training_" if is_training else "",
         split): len(example_list)
    })
    num_per_shard = int(math.ceil(len(example_list) / float(config.num_shards)))
    shard_id_list = list(range(config.num_shards))
    with Manager() as manager:
        shared_example_list = manager.list(example_list)
        shared_subt = manager.dict(subt)
        func = partial(create_one_tfrecord,
                       split,
                       modality,
                       3,
                       shared_example_list,
                       shared_subt,
                       is_training)
        with Pool(8) as p, tqdm(total=config.num_shards, desc="Write tfrecords") as pbar:
            for i, _ in enumerate(p.imap_unordered(func, shard_id_list)):
                pbar.update()


def test():
    config_ = MovieQAConfig()
    TFRECORD_PATTERN = du.FILE_PATTERN.replace('%05d-of-', '*')
    TFRECORD_PATTERN = ('training_' if FLAGS.is_training else '') + TFRECORD_PATTERN
    print(TFRECORD_PATTERN % (config_.dataset_name, 'train', 'fixed_num', config_.num_shards))
    file_names = glob(join(config_.dataset_dir,
                           TFRECORD_PATTERN % (config_.dataset_name, 'train', 'fixed_num', config_.num_shards)))

    file_name_queue = tf.train.string_input_producer(file_names)
    reader = tf.TFRecordReader()
    _, example = reader.read(file_name_queue)
    context_features, sequence_features = du.qa_feature_parsed()
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            q, ql, al, a = sess.run([
                context_parsed['feat'],
                context_parsed['subt'],
                context_parsed['subt_length'],
                context_parsed['ques'],
                context_parsed['ques_length'],
                context_parsed['ans_length'],
                sequence_parsed['ans']])
            print(q, ql, al, a, sep='\n')
        coord.request_stop()
        coord.join(threads)


def main(_):
    if FLAGS.test:
        print('Test tfrecords.')
        test()
    else:
        encode_qa = du.load_json(config.avail_encode_qa_file)
        encode_subtitle = du.load_json(config.encode_subtitle_file)
        print('Json file loading done !!')
        du.exist_make_dirs(config.dataset_dir)
        create_tfrecord(encode_qa['encode_qa_%s' % FLAGS.split],
                        encode_subtitle,
                        split=FLAGS.split,
                        modality=FLAGS.modality,
                        is_training=FLAGS.is_training)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("split", "train", "")
    flags.DEFINE_string("modality", "fixed_num", "")
    flags.DEFINE_bool("is_training", True, "")
    flags.DEFINE_bool("test", False, "")
    FLAGS = flags.FLAGS
    tf.app.run()

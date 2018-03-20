import argparse
from functools import partial
from multiprocessing import Pool, Manager
from os.path import join

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath
from input import find_max_length

_mp = MovieQAPath()
dataset_dir = _mp.dataset_dir


class Args(object):
    def __init__(self):
        pass


def pad_list_numpy(sents, shape):
    if len(shape) > 1:
        arr = np.zeros((1,) + shape, dtype=np.int64)
        for idx, item in enumerate(sents):
            arr[idx][:len(item)] = item
    else:
        arr = np.zeros((1, 1) + shape, dtype=np.int64)
        arr[0][:len(sents)] = sents

    return arr


def create_one_tfrecord(qa, encode_subt, args, video_data, key):
    # num_shards = int(math.ceil(len(qa) / float(args.num_per_shards)))
    print(len(qa))

    if 'subt' not in args.mode:
        output_filename = join(dataset_dir,
                               'feat-%s-%s.tfrecord' % (args.split, key))
    elif 'feat' not in args.mode:
        output_filename = join(dataset_dir,
                               'subt-%s-%s.tfrecord' % (args.split, key))
    else:
        output_filename = join(dataset_dir, '%s-%s.tfrecord' % (args.split, key))

    fu.safe_remove(output_filename)

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        video_list = sorted(list(video_data[key].keys()))
        num_frame = sum([video_data[key][v]['real_frames'] for v in video_list])

        spectrum = np.zeros((len(qa[key]), num_frame), dtype=np.int64)
        ques = np.zeros((len(qa[key]), args.q_max), dtype=np.int64)
        ans = np.zeros((len(qa[key]), 5, args.a_max), dtype=np.int64)
        ql = np.zeros(len(qa[key]), dtype=np.int64)
        al = np.zeros((len(qa[key]), 5), dtype=np.int64)
        gt = []

        for idx, ins in enumerate(qa[key]):
            ques[idx][:len(ins['question'])] = ins['question']
            ql[idx] = len(ins['question'])
            for i in range(5):
                ans[idx][i][:len(ins['answers'][i])] = ins['answers'][i]
                al[idx][i] = len(ins['answers'])
            index = 0
            for v in video_list:
                if v in ins['video_clips']:
                    spectrum[idx][index:index + len(video_data[key][v]['real_frames'])] = 1
                index += video_data[key][v]['real_frames']
            if args.split == 'train' or args.split == 'val':
                gt.append(ins['correct_index'])

        sequence_dict = {
            "ques": du.feature_list(ques, 'int'),
            "ans": du.feature_list(np.reshape(ans, [len(qa[key]), 5 * args.a_max]), 'int'),
            "al": du.feature_list(al, 'int'),
            "spec": du.feature_list(spectrum, 'int'),
            "ql": du.feature_list(ql, 'int'),
        }
        feature = {"N": du.feature(num_frame, 'int'),
                   "N_q": du.feature(len(qa[key]), 'int')}

        if 'subt' in args.mode:
            subt = np.zeros((0, args.subt_max), dtype=np.int64)
            sl = []
            videos = sorted(list(encode_subt[key].keys()))

            for v in videos:
                if encode_subt[key][v]:
                    subt = np.concatenate([subt, du.pad_list_numpy(encode_subt[key][v], args.subt_max)])
                    sl += [len(sent) for sent in encode_subt[key][v]]

            sequence_dict['subt'] = du.feature_list(subt, 'int')
            sequence_dict['sl'] = du.feature_list(sl, 'int')

        if 'feat' in args.mode:
            feat = np.reshape(np.load(join(_mp.feature_dir, key + '.npy')), [-1, 4 * 4 * 1536])
            sequence_dict['feat'] = du.feature_list(feat, 'float')

        if args.split == 'train' or args.split == 'val':
            sequence_dict['gt'] = du.feature_list(gt, 'int')

        feature_lists = tf.train.FeatureLists(feature_list=sequence_dict)
        context = tf.train.Features(feature=feature)
        example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
        tfrecord_writer.write(example.SerializeToString())

    # start_ndx = shard_id * args.num_per_shards
    # end_ndx = min((shard_id + 1) * args.num_per_shards, len(qa))
    #
    # with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    #     for idx in range(start_ndx, end_ndx):
    #         ins = qa[idx]
    #         ques = du.pad_list_numpy(ins['question'], args.q_max)
    #         ans = du.pad_list_numpy(ins['answers'], args.a_max)
    #
    #         if 'subt' in args.mode:
    #             subt = np.zeros((0, args.subt_max), dtype=np.int64)
    #             sl = []
    #         if 'feat' in args.mode:
    #             feat = np.zeros((0, 64 * 1536), dtype=np.float32)
    #
    #         video_clips = sorted(ins['video_clips'])
    #         for v in video_clips:
    #             base_name = fu.basename_wo_ext(v)
    #             v_subt = encode_subt[ins['imdb_key']][base_name]
    #             ds = list(range(0, len(v_subt), 3))
    #             if 'subt' in args.mode:
    #                 subt = np.concatenate([subt, du.pad_list_numpy(v_subt, args.subt_max)[ds]])
    #                 sl += [len(v_subt[i]) for i in ds]
    #             if 'feat' in args.mode:
    #                 feat = np.concatenate([feat, np.reshape(np.load(
    #                     join(_mp.feature_dir, base_name + '.npy'))[ds], [-1, 8 * 8 * 1536])])
    #
    #         sequence_dict = {'ans': du.feature_list(ans, 'int')}
    #
    #         if 'feat' in args.mode:
    #             sequence_dict['feat'] = du.feature_list(feat, 'float')
    #
    #         if 'subt' in args.mode:
    #             sequence_dict['subt'] = du.feature_list(subt, 'int')
    #
    #         feature_lists = tf.train.FeatureLists(feature_list=sequence_dict)
    #
    #         feature = {
    #             "ques": du.feature(ques, 'int'),
    #             "ql": du.feature(len(ins['question']), 'int'),
    #             "al": du.feature([len(a) for a in ins['answers']], 'int')}
    #
    #         if 'subt' in args.mode:
    #             feature["sl"] = du.feature(sl, 'int')
    #
    #         if args.split == 'train' or args.split == 'val':
    #             feature['gt'] = du.feature(ins['correct_index'], 'int')
    #
    #         context = tf.train.Features(feature=feature)
    #         example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    #         tfrecord_writer.write(example.SerializeToString())


def create_tfrecord(encode_qa, encode_subt, split, mode, num_per_shards):
    split_qa = [qa for qa in encode_qa if split in qa['qid']]
    video_data = du.json_load(_mp.video_data_file)
    imdb_split = du.json_load(_mp.splits_file)
    video_split = [k for k in video_data if k in imdb_split[split]]
    qa = {k: [qa for qa in split_qa if qa['imdb_key'] == k] for k in video_split}

    fu.make_dirs(dataset_dir)

    args = Args()

    args.subt_max, args.q_max, args.a_max = find_max_length(encode_qa, encode_subt)

    manager = Manager()

    split_qa = manager.dict(qa)
    encode_subt = manager.dict(encode_subt)
    video_data = manager.dict(video_data)

    args.split = split
    args.mode = mode
    args.num_per_shards = num_per_shards
    func = partial(create_one_tfrecord, split_qa, encode_subt, args, video_data)
    # num_shards = int(math.ceil(len(split_qa) / float(num_per_shards)))
    keys = list(split_qa.keys())
    num_shards = len(keys)
    with Pool(8) as pool, tqdm(total=num_shards, desc='Create %s Tfrecord' % split) as pbar:
        for _ in pool.imap_unordered(func, keys):
            pbar.update()


def count(encode_qa):
    print(len([qa for qa in encode_qa if 'train' in qa['qid']]))
    print(len([qa for qa in encode_qa if 'val' in qa['qid']]))
    print(len([qa for qa in encode_qa if 'test' in qa['qid']]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train/val/test', help='Which split we want to make.')
    parser.add_argument('--num_per_shards', default=32, help='Number of shards.', type=int)
    parser.add_argument('--count', action='store_true', help='Count the number of qa.')
    parser.add_argument('--mode', default='subt+feat', help='Create records with only subtitle.')
    return parser.parse_args()


def main():
    args = parse_args()
    split = args.split

    encode_qa = du.json_load(_mp.encode_qa_file)
    encode_subt = du.json_load(_mp.encode_subtitle_file)

    if args.count:
        count(encode_qa)
    else:
        if 'train' in split:
            create_tfrecord(encode_qa, encode_subt, 'train', args.mode, args.num_per_shards)
        if 'val' in split:
            create_tfrecord(encode_qa, encode_subt, 'val', args.mode, args.num_per_shards)
        if 'test' in split:
            create_tfrecord(encode_qa, encode_subt, 'test', args.mode, args.num_per_shards)


if __name__ == '__main__':
    main()

import argparse
import math
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


def create_one_tfrecord(qa, encode_subt, args, shard_id):
    num_shards = int(math.ceil(len(qa) / float(args.num_per_shards)))

    if 'subt' not in args.mode:
        output_filename = join(dataset_dir,
                               'feat-%s-%04d-of-%04d.tfrecord' % (args.split, shard_id + 1, num_shards))
    elif 'feat' not in args.mode:
        output_filename = join(dataset_dir,
                               'subt-%s-%04d-of-%04d.tfrecord' % (args.split, shard_id + 1, num_shards))
    else:
        output_filename = join(dataset_dir, '%s-%04d-of-%04d.tfrecord' % (args.split, shard_id + 1, num_shards))

    fu.safe_remove(output_filename)
    start_ndx = shard_id * args.num_per_shards
    end_ndx = min((shard_id + 1) * args.num_per_shards, len(qa))

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for idx in range(start_ndx, end_ndx):
            ins = qa[idx]
            ques = du.pad_list_numpy(ins['question'], args.q_max)
            ans = du.pad_list_numpy(ins['answers'], args.a_max)

            if 'subt' in args.mode:
                subt = np.zeros((0, args.subt_max), dtype=np.int64)
                sl = []
            if 'feat' in args.mode:
                feat = np.zeros((0, 64 * 1536), dtype=np.float32)

            video_clips = sorted(ins['video_clips'])
            for v in video_clips:
                base_name = fu.basename_wo_ext(v)
                v_subt = encode_subt[ins['imdb_key']][base_name]
                ds = list(range(0, len(v_subt), 3))
                if 'subt' in args.mode:
                    subt = np.concatenate([subt, du.pad_list_numpy(v_subt, args.subt_max)[ds]])
                    sl += [len(v_subt[i]) for i in ds]
                if 'feat' in args.mode:
                    feat = np.concatenate([feat, np.reshape(np.load(
                        join(_mp.feature_dir, base_name + '.npy'))[ds], [-1, 8 * 8 * 1536])])

            sequence_dict = {'ans': du.feature_list(ans, 'int')}

            if 'feat' in args.mode:
                sequence_dict['feat'] = du.feature_list(feat, 'float')

            if 'subt' in args.mode:
                sequence_dict['subt'] = du.feature_list(subt, 'int')

            feature_lists = tf.train.FeatureLists(feature_list=sequence_dict)

            feature = {
                "ques": du.feature(ques, 'int'),
                "ql": du.feature(len(ins['question']), 'int'),
                "al": du.feature([len(a) for a in ins['answers']], 'int')}

            if 'subt' in args.mode:
                feature["sl"] = du.feature(sl, 'int')

            if args.split == 'train' or args.split == 'val':
                feature['gt'] = du.feature(ins['correct_index'], 'int')

            context = tf.train.Features(feature=feature)
            example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
            tfrecord_writer.write(example.SerializeToString())


def create_tfrecord(encode_qa, encode_subt, split, mode, num_per_shards):
    split_qa = [qa for qa in encode_qa if split in qa['qid']]

    fu.make_dirs(dataset_dir)

    args = Args()

    args.subt_max, args.q_max, args.a_max = find_max_length(encode_qa, encode_subt)

    manager = Manager()

    split_qa = manager.list(split_qa)
    encode_subt = manager.dict(encode_subt)

    args.split = split
    args.mode = mode
    args.num_per_shards = num_per_shards
    func = partial(create_one_tfrecord, split_qa, encode_subt, args)
    num_shards = int(math.ceil(len(split_qa) / float(num_per_shards)))
    with Pool(8) as pool, tqdm(total=num_shards, desc='Create %s Tfrecord' % split) as pbar:
        for _ in pool.imap_unordered(func, list(range(num_shards))):
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

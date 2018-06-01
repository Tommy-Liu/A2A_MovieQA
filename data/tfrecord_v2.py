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

_mp = MovieQAPath()
dataset_dir = _mp.dataset_dir


class Args(object):
    def __init__(self):
        pass


def find_max_length(qa):
    q_max, a_max = 0, 0
    for ins in qa:
        if q_max < len(ins['question']):
            q_max = len(ins['question'])
        for a in ins['answers']:
            if a_max < len(a):
                a_max = len(a)

    return q_max, a_max


def create_one_tfrecord(qa, args, video_data, shard_id):
    num_shards = int(math.ceil(len(qa) / float(args.num_per_shards)))
    start_ndx = shard_id * args.num_per_shards
    end_ndx = min((shard_id + 1) * args.num_per_shards, len(qa))
    output_filename = join(dataset_dir, '%s-%d-of-%d.tfrecord' % (args.split, shard_id + 1, num_shards))

    fu.safe_remove(output_filename)

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for idx in range(start_ndx, end_ndx):
            ins = qa[idx]
            ques = du.pad_list_numpy(ins['question'], args.q_max)
            ans = du.pad_list_numpy(ins['answers'], args.a_max)

            video_list = sorted(list(video_data[ins['imdb_key']].keys()))

            num_frame = sum([int(math.ceil(video_data[ins['imdb_key']][v]['real_frames'] / 15))
                             for v in video_list])
            spectrum = np.zeros(num_frame, dtype=np.int64)

            index = 0
            for v in video_list:
                num = int(math.ceil(video_data[ins['imdb_key']][v]['real_frames'] / 15))
                if v in ins['video_clips']:
                    spectrum[idx][index:(index + num)] = 1
                index += num

            feature_lists = tf.train.FeatureLists(feature_list={
                "ans": du.feature_list(ans, 'int'),
                "spec": du.feature_list(spectrum, 'int')
            })

            feature = {
                "ques": du.feature(ques, 'int'),
                "ql": du.feature(len(ins['question']), 'int'),
                "al": du.feature([len(a) for a in ins['answers']], 'int'),
                "subt": du.feature(join(_mp.encode_dir, ins['imdb_key'] + '.npy').encode(), 'string'),
                "feat": du.feature(join(_mp.feature_dir, ins['imdb_key'] + '.npy').encode(), 'string')
            }

            # if 'subt' in args.mode:
            #     feature['subt'] = du.feature(join(_mp.encode_dir, ins['imdb_key'] + '.npz').encode(), 'string')

            # if 'feat' in args.mode:
            #     feature['feat'] = du.feature(join(_mp.feature_dir, ins['imdb_key'] + '.npy').encode(), 'string')

            if args.split == 'train' or args.split == 'val':
                feature['gt'] = du.feature(ins['correct_index'], 'int')

            context = tf.train.Features(feature=feature)
            example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
            tfrecord_writer.write(example.SerializeToString())


def create_tfrecord(encode_qa, split, mode, num_per_shards):
    split_qa = [qa for qa in encode_qa if split in qa['qid']]

    fu.make_dirs(dataset_dir)

    args = Args()

    args.q_max, args.a_max = find_max_length(encode_qa)

    manager = Manager()

    split_qa = manager.list(split_qa)
    video_data = manager.dict(du.json_load(_mp.video_data_file))

    args.split = split
    args.mode = mode
    args.num_per_shards = num_per_shards
    func = partial(create_one_tfrecord, split_qa, args, video_data)
    num_shards = int(math.ceil(len(split_qa) / float(num_per_shards)))

    with Pool(4) as pool, tqdm(total=num_shards, desc='Create %s Tfrecord' % split) as pbar:
        for _ in pool.imap_unordered(func, list(range(num_shards))):
            pbar.update()


def count(encode_qa):
    print(len([qa for qa in encode_qa if 'train' in qa['qid']]))
    print(len([qa for qa in encode_qa if 'val' in qa['qid']]))
    print(len([qa for qa in encode_qa if 'tests' in qa['qid']]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train/val/tests', help='Which split we want to make.')
    parser.add_argument('--num_per_shards', default=32, help='Number of shards.', type=int)
    parser.add_argument('--count', action='store_true', help='Count the number of qa.')
    parser.add_argument('--mode', default='subt+feat', help='Create records with only subtitle.')
    return parser.parse_args()


def main():
    args = parse_args()
    split = args.split

    encode_qa = du.json_load(_mp.encode_qa_file)

    if args.count:
        count(encode_qa)
    else:
        if 'train' in split:
            create_tfrecord(encode_qa, 'train', args.mode, args.num_per_shards)
        if 'val' in split:
            create_tfrecord(encode_qa, 'val', args.mode, args.num_per_shards)
        if 'tests' in split:
            create_tfrecord(encode_qa, 'tests', args.mode, args.num_per_shards)


if __name__ == '__main__':
    main()

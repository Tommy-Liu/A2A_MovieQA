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


class Args(object):
    def __init__(self):
        pass


def create_one_tfrecord(qa, encode_subt, args, shard_id):
    output_filename = join(
        _mp.dataset_dir, args.mode + '-%04d-of-%04d.tfrecord' % (shard_id + 1, args.num_shards))
    fu.safe_remove(output_filename)

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        # for i in range(start_ndx, end_ndx):
        ins = qa[shard_id]
        ques = du.pad_list_numpy(ins['question'], args.q_max)
        ans = du.pad_list_numpy(ins['answers'], args.a_max)

        subt = np.zeros((0, args.subt_max), dtype=np.int64)
        feat = np.zeros((0, 8 * 8 * 1536), dtype=np.float32)

        sl = []
        for v in ins['video_clips']:
            base_name = fu.basename_wo_ext(v)
            v_subt = encode_subt[ins['imdb_key']][base_name]
            ds = np.arange(0, len(v_subt), 3)
            subt = np.concatenate([
                subt, du.pad_list_numpy(v_subt, args.subt_max)[ds]])
            sl += [len(v_subt[i]) for i in range(0, len(v_subt), 3)]
            feat = np.concatenate([
                feat, np.reshape(np.load(join(_mp.feature_dir, base_name + '.npy'))[ds], [-1, 8 * 8 * 1536])])

        feature_lists = tf.train.FeatureLists(feature_list={
            "subt": du.feature_list(subt, 'int'),
            "feat": du.feature_list(feat, 'float'),
            "ans": du.feature_list(ans, 'int'),
        })

        feature = {
            "ques": du.feature(ques, 'int'),
            "ql": du.feature(len(ins['question']), 'int'),
            "al": du.feature([len(a) for a in ins['answers']], 'int'),
            "sl": du.feature(sl, 'int')}

        if args.mode == 'train' or args.mode == 'val':
            feature['gt'] = du.feature(ins['correct_index'], 'int')

        context = tf.train.Features(feature=feature)
        example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
        tfrecord_writer.write(example.SerializeToString())


def create_tfrecord(encode_qa, encode_subt, split):
    split_qa = [qa for qa in encode_qa if split in qa['qid']]

    # print(train_qa[262])
    fu.make_dirs(_mp.dataset_dir)

    args = Args()

    args.subt_max, args.q_max, args.a_max = find_max_length(encode_qa, encode_subt)

    manager = Manager()

    split_qa = manager.list(split_qa)
    encode_subt = manager.dict(encode_subt)

    args.mode = split
    args.num_shards = len(split_qa)
    func = partial(create_one_tfrecord, split_qa, encode_subt, args)

    with Pool(8) as pool, tqdm(total=args.num_shards, desc='Create Tfrecord') as pbar:
        for _ in pool.imap_unordered(func, list(range(args.num_shards))):
            pbar.update()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train/val/test', help='Which split we want to make.')
    return parser.parse_args().split


def main():
    split = parse_args()
    encode_qa = du.json_load(_mp.encode_qa_file)
    encode_subt = du.json_load(_mp.encode_subtitle_file)

    if 'train' in split:
        create_tfrecord(encode_qa, encode_subt, 'train')
    if 'val' in split:
        create_tfrecord(encode_qa, encode_subt, 'val')
    if 'test' in split:
        create_tfrecord(encode_qa, encode_subt, 'test')


if __name__ == '__main__':
    main()

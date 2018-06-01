import math
import os
import textwrap
from functools import partial
from multiprocessing import Pool, Manager
from os.path import join

import matplotlib

matplotlib.use('Agg')
import numpy as np
import seaborn as sns

sns.set()
from tqdm import tqdm
import argparse
from config import MovieQAPath
from data.data_loader import Subtitle

import matplotlib.pyplot as plt
import utils.data_utils as du
import utils.func_utils as fu

_mp = MovieQAPath()


def l2_norm(x, axis=1, eps=1e-6):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def do_attn(index, subtitle, counter, qa):
    qa_embed = np.load(join(_mp.encode_dir, qa['qid'] + '.npy'))
    ques_embed, ans_embed = l2_norm(qa_embed[[0]]), l2_norm(qa_embed[1:])
    subt_embed = np.load(join(_mp.encode_dir, qa['imdb_key'] + '.npy'))
    spec = np.load(join(_mp.encode_dir, qa['qid'] + '_spec' + '.npy'))
    subt_embed = l2_norm(subt_embed[spec == 1])

    sq = np.matmul(subt_embed, ques_embed.transpose())
    sa = np.matmul(subt_embed, ans_embed.transpose())

    sqa = np.expand_dims(sq + sa, axis=0)
    sqa = np.transpose(sqa, [2, 1, 0])
    sqas = np.expand_dims(subt_embed, axis=0) * sqa
    sqas = np.sum(sqas, axis=1)
    sqas = l2_norm(sqas)

    output = np.sum(sqas * ans_embed, axis=1)
    choice = np.argmax(output)

    counter.value += int(int(choice) == qa['correct_index'])

    if args.img:
        iid = [idx for i, idx in enumerate(index[qa['imdb_key']]) if spec[i] == 1]
        sentences = [textwrap.fill(subtitle[qa['imdb_key']]['lines'][idx], 40) for idx in iid]

        attn = np.concatenate([
            np.matmul(subt_embed, ques_embed.transpose()),
            np.matmul(subt_embed, ans_embed.transpose())
        ], axis=1)

        qa_cor = np.matmul(ques_embed, ans_embed.transpose()).squeeze()

        h = attn.shape[0]
        num = int(math.ceil(h / 30))

        for i in range(num):
            s, e = i * 30, min((i + 1) * 30, h)
            fig = plt.figure(figsize=(12, int(math.ceil((e - s) / 2))))
            ax = sns.heatmap(attn[s:e])
            ax.set_xticklabels([textwrap.fill(qa['question'], 40)] +
                               [textwrap.fill(ans + ' %.4f' % qa_cor[idx], 40) for idx, ans in
                                enumerate(qa['answers'])], rotation=45, ha='right')
            temp_sentences = sentences[s:e]
            temp_sentences.reverse()
            ax.set_yticklabels(temp_sentences, rotation=0)
            ax.set_title('%d %d' % (qa['correct_index'], int(choice)))
            fig.savefig(os.path.join(stat_dir, '%s_%d.jpg') % (qa['qid'].replace(':', ''), i), bbox_inches='tight')
            plt.close()


def main():
    video_qa = [qa for qa in du.json_load(_mp.qa_file) if qa['video_clips']]
    train = [qa for qa in video_qa if 'train' in qa['qid']]
    val = [qa for qa in video_qa if 'val' in qa['qid']]
    test = [qa for qa in video_qa if 'tests' in qa['qid']]

    with Pool(4) as pool, Manager() as manager:
        index = manager.dict(du.json_load(_mp.sample_index_file))
        subtitle = manager.dict(Subtitle().get())
        counter = manager.Value(int, 0)
        func = partial(do_attn, index, subtitle, counter)
        for _ in pool.imap_unordered(func, tqdm(train)):
            pass
        print('train acc: %.4f' % (counter.value / len(train)))
        counter = manager.Value(int, 0)
        func = partial(do_attn, index, subtitle, counter)
        for _ in pool.imap_unordered(func, tqdm(val)):
            pass
        print('val acc: %.4f' % (counter.value / len(val)))


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', action='store_true', help='Create attention images.')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    if args.img:
        stat_dir = os.path.join('stat')
        if os.path.exists(stat_dir):
            os.system('rm -rf %s' % stat_dir)
        fu.make_dirs(stat_dir)
    main()

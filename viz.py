import argparse
import math
import os
import textwrap
from functools import partial

import matplotlib
import numpy as np
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)
matplotlib.use('Agg')

import utils.func_utils as fu
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from multiprocessing import Manager, Pool
# import utils.func_utils as fu
import utils.data_utils as du

from config import MovieQAPath
from data.data_loader import QA, Subtitle

_mp = MovieQAPath()


def gen_attn(attn, pair, index, subtitle, qa):
    attn = attn[qa['qid'].replace(':', '')]
    pair = pair[qa['qid'].replace(':', '')][:]
    spec = np.load(os.path.join(_mp.encode_dir, qa['qid'] + '_spec' + '.npy'))
    iid = [idx for i, idx in enumerate(index[qa['imdb_key']]) if spec[i] == 1]
    sentences = [textwrap.fill(subtitle[qa['imdb_key']]['lines'][idx], 40) for idx in iid]
    h = attn.shape[0]
    num = int(math.ceil(h / 30))

    for i in range(num):
        s, e = i * 30, min((i + 1) * 30, h)
        fig = plt.figure(figsize=(6, int(math.ceil((e - s) / 2))))
        ax = sns.heatmap(attn[s:e], vmin=0, vmax=1)
        ax.set_xticklabels([textwrap.fill(qa['question'], 40)] +
                           [textwrap.fill(ans, 40) for ans in qa['answers']], rotation=45, ha='right')
        temp_sentences = sentences[s:e]
        temp_sentences.reverse()
        ax.set_yticklabels(temp_sentences, rotation=0)
        ax.set_title('%d %d' % (pair[0], pair[1]))
        fig.savefig(os.path.join(fig_dir, '%s_%d.jpg') % (qa['qid'].replace(':', ''), i), bbox_inches='tight')
        plt.close()


def main():
    with Manager() as manager, Pool(4) as pool:
        print('Loading data...')
        index = manager.dict(du.json_load(_mp.sample_index_file))
        subtitle = manager.dict(Subtitle().get())
        train_attn = manager.dict(dict(
            np.load(os.path.join(_mp.attn_dir, args.mod, 'train_attn.npz'))))
        val_attn = manager.dict(dict(
            np.load(os.path.join(_mp.attn_dir, args.mod, 'val_attn.npz'))))
        train_pair = manager.dict(dict(
            np.load(os.path.join(_mp.attn_dir, args.mod, 'train_pair.npz'))))
        val_pair = manager.dict(dict(
            np.load(os.path.join(_mp.attn_dir, args.mod, 'val_pair.npz'))))
        train_qa = QA().include(video_clips=True, split={'train'}).get()
        val_qa = QA().include(video_clips=True, split={'val'}).get()
        print('Loading done!')
        func = partial(gen_attn, train_attn, train_pair, index, subtitle)
        for _ in pool.imap_unordered(func, tqdm([train_qa[i] for i in range(0, len(train_qa), 1)])):
            pass
        func = partial(gen_attn, val_attn, val_pair, index, subtitle)
        for _ in pool.imap_unordered(func, tqdm([val_qa[i] for i in range(0, len(val_qa), 1)])):
            pass


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod', default='model_se_spec3-02-', help='Model used to train.')
    parser.add_argument('--reset', action='store_true', help='Remove the attention images.')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    fig_dir = os.path.join('fig', args.mod)
    if args.reset and os.path.exists(fig_dir):
        os.system('rm -rf %s' % fig_dir)
    fu.make_dirs(fig_dir)
    main()

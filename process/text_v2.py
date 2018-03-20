import os
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from multiprocessing import Manager, Pool
from os.path import exists, join

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath
from data.data_loader import Subtitle, FrameTime, duration, QA
from embed.args import EmbeddingPath

_mp = MovieQAPath()
_ep = EmbeddingPath()


def binary_search(s, e, t):
    # Find last one smaller than t in a sorted array.
    lower = 0
    upper = len(s) - 1
    pivot = (lower + upper) // 2
    while lower < upper:
        if s[pivot] > t:
            upper = pivot - 1
        else:
            if e[pivot] >= t:
                return pivot
            else:
                lower = pivot + 1
        pivot = (lower + upper) // 2
    return pivot


def align_subtitle(video_data, frame_time, subtitle, key):
    subt = subtitle[key]
    ft = frame_time[key]
    video_clips = sorted(list(video_data[key].keys()))
    subt_embedding = np.zeros((0, 300), dtype=np.float32)

    for video in video_clips:
        start_frame, end_frame = duration(video)
        subsample_list = list(range(0, video_data[key][video]['real_frames'], 15))
        temp_embedding = np.zeros((len(subsample_list), 300), dtype=np.float32)

        for idx, i in enumerate(subsample_list):
            time = ft[min(start_frame + i, len(ft) - 1)]
            index = binary_search(subt['start'], subt['end'], time)
            if subt['start'][index] <= time <= subt['end'][index]:
                temp_embedding[idx] = subt['lines'][index]

        subt_embedding = np.concatenate([subt_embedding, temp_embedding], axis=0)

    np.save(join(_mp.encode_dir, key + '.npy'), subt_embedding)


def subtitle_process(video_data, frame_time, subtitle):
    if not exists(_mp.encode_dir):
        fu.make_dirs(_mp.encode_dir)

    manager = Manager()
    video_data = manager.dict(video_data)
    frame_time = manager.dict(frame_time)
    subtitle = manager.dict(subtitle)

    keys = list(video_data.keys())
    align_func = partial(align_subtitle, video_data, frame_time, subtitle)

    with Pool(4) as p, tqdm(total=len(keys), desc="Align subtitle") as pbar:
        for _ in p.imap_unordered(align_func, keys):
            pbar.update()


def load_frequency():
    freq = {}
    with open(_mp.freq_file, 'r') as f:
        for l in f:
            w, p = l.rsplit(' ')
            freq[w] = int(p)

    return freq


def collect_embedding(qa, subtitle, video_data, filter_vocab, vocab_embed, frequency):
    total = sum(list(frequency.values()))
    frequency = {k: v / total for k, v in frequency.items()}
    all_embedding = np.zeros((0, 300), dtype=np.float32)

    for key in tqdm(video_data, desc='Create Sentence Embedding'):
        subt = subtitle[key]
        sent_embedding = np.zeros((len(subt['lines']), 300), dtype=np.float32)
        for idx, line in enumerate(subt['lines']):
            embed = vocab_embed[[filter_vocab[w] for w in line]]
            w = 10 ** (-3) / (10 ** (-3) + np.expand_dims(np.array([frequency[w] for w in line]), axis=1))
            sent = np.mean(embed * w, axis=0)
            sent_embedding[idx] = sent
        subt['lines'] = sent_embedding
        all_embedding = np.concatenate([all_embedding, sent_embedding], axis=0)

    for ins in tqdm(qa, desc='Create QA Embedding'):
        embed = vocab_embed[[filter_vocab[w] for w in ins['question'] if w in filter_vocab]]
        w = 10 ** (-3) / (10 ** (-3) + np.expand_dims(np.array([frequency[w] for w in ins['question']]), axis=1))
        sent = np.mean(embed * w, axis=0, keepdims=True)
        ins['question'] = sent
        all_embedding = np.concatenate([all_embedding, ins['question']], axis=0)
        ans_embedding = np.zeros((5, 300), dtype=np.float32)
        for idx, candidate in enumerate(ins['answers']):
            embed = vocab_embed[[filter_vocab[w] for w in candidate]]
            w = 10 ** (-3) / (10 ** (-3) + np.expand_dims(np.array([frequency[w] for w in candidate]), axis=1))
            sent = np.mean(embed * w, axis=0)
            ans_embedding[idx] = sent
        ins['answers'] = ans_embedding
        all_embedding = np.concatenate([all_embedding, ins['answers']], axis=0)

    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(all_embedding)

    for key in tqdm(video_data, desc='Subtitle Remove Component'):
        subtitle[key]['lines'] = subtitle[key]['lines'] - \
                                 subtitle[key]['lines'].dot(np.transpose(svd.components_)) * svd.components_
        norm = np.linalg.norm(subtitle[key]['lines'], axis=1, keepdims=True)
        norm = np.select([norm > 0], [norm], default=1.)
        subtitle[key]['lines'] = subtitle[key]['lines'] / norm

    for ins in tqdm(qa, desc='QA Remove Component'):
        ins['question'] = ins['question'] - ins['question'].dot(np.transpose(svd.components_)) * svd.components_
        norm = np.linalg.norm(ins['question'], axis=1, keepdims=True)
        norm = np.select([norm > 0], [norm], default=1.)
        ins['question'] = ins['question'] / norm

        ins['answers'] = ins['answers'] - ins['answers'].dot(np.transpose(svd.components_)) * svd.components_
        norm = np.linalg.norm(ins['answers'], axis=1, keepdims=True)
        norm = np.select([norm > 0], [norm], default=1.)
        ins['answers'] = ins['answers'] / norm

        np.save(join(_mp.encode_dir, ins['qid'] + '.npy'), np.concatenate([ins['question'], ins['answers']], axis=0))


def create_vocab(qa, subtitle, video_data, gram_vocab, gram_embed):
    if not exists(_mp.vocab_file):
        vocab = Counter()

        for key in tqdm(video_data, desc='Tokenize Subtitle'):
            subt = subtitle[key]
            for idx, line in enumerate(subt['lines']):
                line = word_tokenize(line.strip().lower())
                vocab.update(line)
                subt['lines'][idx] = line

        for ins in tqdm(qa, desc='Tokenize QA'):
            ins['question'] = word_tokenize(ins['question'].strip().lower())
            vocab.update(ins['question'])
            ins['answers'] = [word_tokenize(sent.strip().lower()) if sent else ['.']
                              for sent in ins['answers']]
            for sent in ins['answers']:
                vocab.update(sent)

        filter_vocab, idx_vocab = {}, 1

        frequency = {}
        vocab_embed = np.zeros((len(vocab) + 1, gram_embed.shape[1]), dtype=np.float32)
        for v in vocab:
            v_ = '<' + v + '>'
            v_gram = [c for c in v_] + [v_[i:i + 3] for i in range(len(v_) - 2)] + \
                     [v_[i:i + 6] for i in range(len(v_) - 5)]
            v_gram_code = [gram_vocab[gram] for gram in v_gram if gram in gram_vocab]
            if v_gram_code:
                frequency[v] = vocab[v]
                filter_vocab[v] = idx_vocab
                vocab_embed[idx_vocab] = np.sum(gram_embed[v_gram_code], axis=0)
                idx_vocab += 1
        vocab_embed = vocab_embed[:(len(filter_vocab) + 1)]
        du.json_dump(frequency, _mp.freq_file)
        du.json_dump(filter_vocab, _mp.vocab_file)
        np.save(_mp.embedding_file, vocab_embed)
        print(len(vocab_embed))
        du.json_dump(subtitle, _mp.temp_subtitle_file)
        du.json_dump(qa, _mp.tokenize_qa)
    else:
        filter_vocab = du.json_load(_mp.vocab_file)
        subtitle = du.json_load(_mp.temp_subtitle_file)
        vocab_embed = np.load(_mp.embedding_file)
        frequency = du.json_load(_mp.freq_file)
        qa = du.json_load(_mp.tokenize_qa)

    return filter_vocab, subtitle, vocab_embed, frequency, qa


def remove_all():
    fu.safe_remove(_mp.tokenize_qa)
    fu.safe_remove(_mp.tokenize_subt)
    fu.safe_remove(_mp.vocab_file)
    fu.safe_remove(_mp.embedding_file)
    fu.safe_remove(_mp.encode_subtitle_file)
    fu.safe_remove(_mp.encode_qa_file)
    fu.safe_remove(_mp.freq_file)
    if exists(_mp.encode_dir):
        os.system('rm -rf %s' % _mp.encode_dir)


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--rm', action='store_true', help='Remove pre-processing files.')
    parser.add_argument('--max', action='store_true', help='Find maximal length of all input.')
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.rm:
        remove_all()

    qa = QA().include(video_clips=True).get()
    video_data = du.json_load(_mp.video_data_file)
    frame_time = FrameTime().get()
    subtitle = Subtitle().get()
    gram_vocab = {k: i for i, k in enumerate(du.json_load(_ep.gram_vocab_file))}
    gram_embed = np.load(_ep.gram_embedding_vec_file)

    filter_vocab, subtitle, vocab_embed, frequency, qa = \
        create_vocab(qa, subtitle, video_data, gram_vocab, gram_embed)

    collect_embedding(qa, subtitle, video_data, filter_vocab, vocab_embed, frequency)

    subtitle_process(video_data, frame_time, subtitle)


if __name__ == '__main__':
    main()

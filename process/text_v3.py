import os
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from multiprocessing import Manager, Pool

import numpy as np
from nltk.tokenize import wordpunct_tokenize
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath
from data.data_loader import Subtitle, FrameTime, duration, QA
from embed.args import EmbeddingPath

_mp = MovieQAPath()
_ep = EmbeddingPath()


def lsbs(a, t):
    # Find last one smaller than t in a sorted array.
    lower = 0
    upper = len(a) - 1
    pivot = int((lower + upper) / 2)
    while lower < upper:
        if a[pivot] > t:
            upper = pivot - 1
        else:
            lower = pivot + 1
        pivot = int((lower + upper) / 2)
    for i in range(max(pivot - 1, 0), min(pivot + 2, len(a) - 1)):
        if a[i] <= t < a[i + 1]:
            return i
    return pivot


def flbs(a, t):
    # Find first one larger than t in a sorted array.
    lower = 0
    upper = len(a) - 1
    pivot = int((lower + upper) / 2)
    while lower < upper:
        if a[pivot] > t:
            upper = pivot - 1
        else:
            lower = pivot + 1
        pivot = int((lower + upper) / 2)
    for i in range(max(pivot - 1, 0), min(pivot + 2, len(a) - 1)):
        if a[i - 1] < t <= a[i]:
            return i
    return pivot


def sample_frame(video_data, frame_time, subtitle, sample, key):
    subt = subtitle[key]
    ft = frame_time[key]
    vd = video_data[key]
    temp_sample = {}
    subt_embedding = []

    for video in sorted(list(vd.keys())):
        start_frame, end_frame = duration(video)
        start_frame = max(start_frame, 0)
        end_frame = min(start_frame + vd[video]['real_frames'], len(ft) - 1)
        start_time, end_time = ft[start_frame], ft[end_frame]

        start_index, end_index = flbs(subt['end'], start_time), lsbs(subt['start'], end_time)

        temp_sample[video] = []
        for i in range(start_index, end_index + 1):
            i_start, i_end = subt['start'][i], subt['end'][i]
            i_start_frame = min(max(flbs(ft, i_start) - start_frame, 0), vd[video]['real_frames'])
            i_end_frame = max(min(lsbs(ft, i_end) - start_frame, vd[video]['real_frames']), 0)
            sample_list = list(range(i_start_frame, i_end_frame, 6))
            temp_sample[video].extend(sample_list)
            subt_embedding.append(np.tile(subt['lines'][i], [len(sample_list), 1]))

    sample[key] = temp_sample
    np.save(os.path.join(_mp.encode_dir, key + '.npy'), np.concatenate(subt_embedding, axis=0))


def sample_frame_v2(video_data, frame_time, subtitle, sample, key):
    subt = subtitle[key]
    ft = frame_time[key]
    vd = video_data[key]
    temp_sample = {}
    subt_embedding = []
    a = 0
    b = 0
    for video in sorted(list(vd.keys())):
        start_frame, end_frame = duration(video)
        start_frame = max(start_frame, 0)
        end_frame = min(start_frame + vd[video]['real_frames'], len(ft) - 1)
        start_time, end_time = ft[start_frame], ft[end_frame]

        start_index, end_index = flbs(subt['end'], start_time), lsbs(subt['start'], end_time)
        # assert start_index <= end_index, '%s index reversed. %d %d\n%f %f %f %f\n%f %f' % \
        #                                  (video, start_index, end_index, subt['start'][start_index], subt['end'][start_index],
        #                                   subt['start'][end_index], subt['end'][end_index],
        #                                   start_time, end_time)
        if start_index > end_index:
            end_index = start_index
        temp_sample[video] = []
        for i in range(start_index, end_index + 1):
            i_start, i_end = subt['start'][i], subt['end'][i]
            i_start_frame = min(max(flbs(ft, i_start) - start_frame, 0), vd[video]['real_frames'])
            i_end_frame = max(min(lsbs(ft, i_end) - start_frame, vd[video]['real_frames']), 0)
            temp_sample[video].append((i_start_frame + i_end_frame) // 2)
            a += 1
        b += len(subt['lines'][start_index:end_index + 1])
        assert a == b, '%s not aligned. %d %d %d %d %d %d' % \
                       (key, a, b, end_index + 1 - start_index, len(subt['lines'][start_index:end_index + 1]),
                        start_index, end_index)
        subt_embedding.append(subt['lines'][start_index:end_index + 1])

    sample[key] = temp_sample
    np.save(os.path.join(_mp.encode_dir, key + '.npy'), np.concatenate(subt_embedding, axis=0))


def subtitle_process(video_data, frame_time, subtitle):
    manager = Manager()
    sample = manager.dict()
    video_data = manager.dict(video_data)
    frame_time = manager.dict(frame_time)
    subtitle = manager.dict(subtitle)

    keys = list(video_data.keys())
    align_func = partial(sample_frame_v2, video_data, frame_time, subtitle, sample)

    with Pool(4) as p, tqdm(total=len(keys), desc="Align subtitle") as pbar:
        for _ in p.imap_unordered(align_func, keys):
            pbar.update()

    du.json_dump(sample.copy(), _mp.sample_frame_file)
    return sample.copy()


def collect_embedding(qa, subtitle, video_data, filter_vocab, vocab_embed, frequency):
    fu.make_dirs(_mp.encode_dir)
    total = sum(list(frequency.values()))
    frequency = {k: v / total for k, v in frequency.items()}
    all_embedding = []

    for key in tqdm(video_data, desc='Create Sentence Embedding'):
        subt = subtitle[key]
        sent_embedding = np.zeros((len(subt['lines']), 300), dtype=np.float32)
        for idx, line in enumerate(subt['lines']):
            embed = vocab_embed[[filter_vocab[w] for w in line]]
            w = 10 ** (-3) / (10 ** (-3) + np.expand_dims(np.array([frequency[w] for w in line]), axis=1))
            sent = np.mean(embed * w, axis=0)
            sent_embedding[idx] = sent
        subt['lines'] = sent_embedding
        all_embedding.append(sent_embedding)

    for ins in tqdm(qa, desc='Create QA Embedding'):
        embed = vocab_embed[[filter_vocab[w] for w in ins['question'] if w in filter_vocab]]
        w = 10 ** (-3) / (10 ** (-3) + np.expand_dims(np.array([frequency[w] for w in ins['question']]), axis=1))
        sent = np.mean(embed * w, axis=0, keepdims=True)
        ins['question'] = sent
        all_embedding.append(ins['question'])
        ans_embedding = np.zeros((5, 300), dtype=np.float32)
        for idx, candidate in enumerate(ins['answers']):
            embed = vocab_embed[[filter_vocab[w] for w in candidate]]
            w = 10 ** (-3) / (10 ** (-3) + np.expand_dims(np.array([frequency[w] for w in candidate]), axis=1))
            sent = np.mean(embed * w, axis=0)
            ans_embedding[idx] = sent
        ins['answers'] = ans_embedding
        all_embedding.append(ins['answers'])

    all_embedding = np.concatenate(all_embedding, axis=0)
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(all_embedding)

    for key in tqdm(video_data, desc='Subtitle Remove Component'):
        subtitle[key]['lines'] = subtitle[key]['lines'] - \
                                 subtitle[key]['lines'].dot(np.transpose(svd.components_)) * svd.components_
        # norm = np.linalg.norm(subtitle[key]['lines'], axis=1, keepdims=True)
        # norm = np.select([norm > 0], [norm], default=1.)
        # subtitle[key]['lines'] = subtitle[key]['lines'] / norm

    for ins in tqdm(qa, desc='QA Remove Component'):
        ins['question'] = ins['question'] - ins['question'].dot(np.transpose(svd.components_)) * svd.components_
        # norm = np.linalg.norm(ins['question'], axis=1, keepdims=True)
        # norm = np.select([norm > 0], [norm], default=1.)
        # ins['question'] = ins['question'] / norm

        ins['answers'] = ins['answers'] - ins['answers'].dot(np.transpose(svd.components_)) * svd.components_
        # norm = np.linalg.norm(ins['answers'], axis=1, keepdims=True)
        # norm = np.select([norm > 0], [norm], default=1.)
        # ins['answers'] = ins['answers'] / norm

        np.save(os.path.join(_mp.encode_dir, ins['qid'] + '.npy'),
                np.concatenate([ins['question'], ins['answers']], axis=0))


def create_vocab(qa, subtitle, video_data, gram_vocab, gram_embed):
    if not os.path.exists(_mp.vocab_file):
        vocab = Counter()

        for key in tqdm(video_data, desc='Tokenize Subtitle'):
            subt = subtitle[key]
            for idx, line in enumerate(subt['lines']):
                line = wordpunct_tokenize(line.strip().lower())
                vocab.update(line)
                subt['lines'][idx] = line

        for ins in tqdm(qa, desc='Tokenize QA'):
            ins['question'] = wordpunct_tokenize(ins['question'].strip().lower())
            vocab.update(ins['question'])
            ins['answers'] = [wordpunct_tokenize(sent.strip().lower()) if sent else ['.']
                              for sent in ins['answers']]
            for sent in ins['answers']:
                vocab.update(sent)

        filter_vocab, idx_vocab = {}, 1

        frequency = {}
        vocab_embed = np.zeros((len(vocab) + 1, gram_embed.shape[1]), dtype=np.float32)
        for v in tqdm(vocab, desc='Create Embedding'):
            v_ = '<' + v + '>'
            v_gram = [c for c in v] + [v_[i:i + 3] for i in range(len(v_) - 2)] + \
                     [v_[i:i + 6] for i in range(len(v_) - 5)]
            v_gram_code = [gram_vocab[gram] for gram in v_gram if gram in gram_vocab]
            if v_gram_code:
                frequency[v] = vocab[v]
                filter_vocab[v] = idx_vocab
                vocab_embed[idx_vocab] = np.sum(gram_embed[v_gram_code], axis=0)
                idx_vocab += 1
        vocab_embed = vocab_embed[:idx_vocab]
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
    # fu.safe_remove(_mp.subtitle_file)
    # fu.safe_remove(_mp.tokenize_qa)
    # fu.safe_remove(_mp.tokenize_subt)
    # fu.safe_remove(_mp.vocab_file)
    # fu.safe_remove(_mp.embedding_file)
    # fu.safe_remove(_mp.freq_file)
    fu.safe_remove(_mp.sample_frame_file)
    if os.path.exists(_mp.encode_dir):
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

    sample = subtitle_process(video_data, frame_time, subtitle)

    for ins in tqdm(qa, desc='Create Spectrum'):
        video_list = sorted(list(video_data[ins['imdb_key']].keys()))

        num_frame = sum([len(sample[ins['imdb_key']][v])
                         for v in video_list])
        spectrum = np.zeros(num_frame, dtype=np.int64)

        index = 0
        for v in video_list:
            num = len(sample[ins['imdb_key']][v])
            if v + '.mp4' in ins['video_clips']:
                spectrum[index:index + num] = 1
            index += num
        assert np.sum(spectrum) > 0, '%s no content needed.' % ins['qid']
        np.save(os.path.join(_mp.encode_dir, ins['qid'] + '_spec' + '.npy'),
                spectrum)


if __name__ == '__main__':
    main()

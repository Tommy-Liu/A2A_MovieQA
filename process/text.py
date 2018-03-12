from argparse import ArgumentParser
from collections import Counter
from functools import partial
from multiprocessing import Pool, Manager
from os.path import exists

import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath
from data.data_loader import Subtitle, FrameTime, duration, QA
from embed.args import EmbeddingPath
from input import find_max_length

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
                lower = pivot
        pivot = (lower + upper) // 2
    return pivot


def align_subtitle(video_data, frame_time, subtitle, tokenize_subt, key):
    subt = subtitle[key]
    ft = frame_time[key]
    video_clips = video_data[key]
    temp_tokenize = {}

    for video in video_clips:
        start_frame, end_frame = duration(video)
        temp_tokenize[video] = []

        for i in range(0, video_clips[video]['real_frames'], 10):
            time = ft[min(start_frame + i, len(ft) - 1)]
            index = binary_search(subt['start'], subt['end'], time)
            if subt['start'][index] <= time <= subt['end'][index] and subt['lines'][index].strip():
                temp_tokenize[video].append(
                    word_tokenize(subt['lines'][index].strip().lower()))
            else:
                temp_tokenize[video].append(['.'])

            if not temp_tokenize[video][-1]:
                temp_tokenize[video][-1] = ['.']

    tokenize_subt[key] = temp_tokenize


def subtitle_process(video_data, frame_time, subtitle):
    if not exists(_mp.tokenize_subt):

        manager = Manager()
        tokenize_subt = manager.dict()
        video_data = manager.dict(video_data)
        frame_time = manager.dict(frame_time)
        subtitle = manager.dict(subtitle)

        keys = list(video_data.keys())
        align_func = partial(align_subtitle, video_data, frame_time, subtitle, tokenize_subt)

        with Pool(16) as p, tqdm(total=len(keys), desc="Align subtitle") as pbar:
            for i, _ in enumerate(p.imap_unordered(align_func, keys)):
                pbar.update()

        res = tokenize_subt.copy()
        du.json_dump(res, _mp.tokenize_subt)

    else:

        res = du.json_load(_mp.tokenize_subt)

    return res


def tokenize_question_answer(qa):
    if not exists(_mp.tokenize_qa):

        tokenize_qa = []

        for ins in tqdm(qa, desc='Tokenize qa'):
            ins['question'] = word_tokenize(ins['question'].strip().lower())
            ins['answers'] = [word_tokenize(sent.strip().lower()) if sent else ['.']
                              for sent in ins['answers']]
            tokenize_qa.append(ins)

        du.json_dump(tokenize_qa, _mp.tokenize_qa)

    else:

        tokenize_qa = du.json_load(_mp.tokenize_qa)

    return tokenize_qa


def create_vocab(tokenize_subt, tokenize_qa):
    vocab = Counter()

    for ins in tqdm(tokenize_qa, desc='Create vocabulary'):
        imdb = ins['imdb_key']
        vocab.update(ins['question'])
        for sent in ins['answers']:
            vocab.update(sent)
        for video in ins['video_clips']:
            video = fu.basename_wo_ext(video)
            for sent in tokenize_subt[imdb][video]:
                vocab.update(sent)

    res = {v: i + 1 for i, v in enumerate(vocab.keys())}
    return res


def create_vocab_embedding(tokenize_subt, tokenize_qa):
    if not exists(_mp.embedding_file) or not exists(_mp.vocab_file):

        vocab = create_vocab(tokenize_subt, tokenize_qa)
        filter_vocab, idx_vocab = {}, 1
        gram_vocab = {k: i for i, k in enumerate(du.json_load(_ep.gram_vocab_file))}
        gram_embed = np.load(_ep.gram_embedding_vec_file)
        vocab_embed = np.zeros((len(vocab) + 1, gram_embed.shape[1]), dtype=np.float32)

        for idx, v in enumerate(tqdm(vocab, desc='Create embedding of vocabulary')):
            v_ = '<' + v + '>'
            v_gram = [c for c in v_] + [v_[i:i + 3] for i in range(len(v_) - 2)] + \
                     [v_[i:i + 6] for i in range(len(v_) - 5)]
            v_gram_code = [gram_vocab[gram] for gram in v_gram if gram in gram_vocab]
            if v_gram_code:
                filter_vocab[v] = idx_vocab
                vocab_embed[idx + 1] = np.sum(gram_embed[v_gram_code], axis=0)
                idx_vocab += 1

        norm = np.linalg.norm(vocab_embed, axis=1, keepdims=True)
        norm = np.select([norm > 0], [norm], default=1.)
        print(norm.shape)
        norm_vocab_embed = vocab_embed / norm
        print(norm_vocab_embed.shape)

        du.json_dump(filter_vocab, _mp.vocab_file)
        np.save(_mp.embedding_file, norm_vocab_embed)
    else:
        filter_vocab = du.json_load(_mp.vocab_file)
    return filter_vocab


def encode_sentence(tokenize_subt, tokenize_qa, vocab):
    if not exists(_mp.encode_subtitle_file) or not exists(_mp.encode_qa_file):

        encode_subt, encode_qa = {}, tokenize_qa
        for imdb in tqdm(tokenize_subt, desc='Encode subtitle'):
            encode_subt[imdb] = {}
            for v in tokenize_subt[imdb]:
                encode_subt[imdb][v] = []
                for sent in tokenize_subt[imdb][v]:
                    temp = [vocab[w] for w in sent if w in vocab]
                    if temp:
                        encode_subt[imdb][v].append(temp)
                    else:
                        encode_subt[imdb][v].append([vocab['.']])

        for ins in tqdm(encode_qa, desc='Encode question answer'):
            temp = [vocab[w] for w in ins['question'] if w in vocab]
            if temp:
                ins['question'] = temp
            else:
                ins['question'] = [vocab['.']]

            for idx, a in enumerate(ins['answers']):
                temp = [vocab[w] for w in a if w in vocab]
                if temp:
                    ins['answers'][idx] = temp
                else:
                    ins['answers'][idx] = [vocab['.']]

        du.json_dump(encode_subt, _mp.encode_subtitle_file)
        du.json_dump(encode_qa, _mp.encode_qa_file)


def remove_all():
    fu.safe_remove(_mp.tokenize_qa)
    fu.safe_remove(_mp.tokenize_subt)
    fu.safe_remove(_mp.vocab_file)
    fu.safe_remove(_mp.embedding_file)
    fu.safe_remove(_mp.encode_subtitle_file)
    fu.safe_remove(_mp.encode_qa_file)


def print_max():
    encode_subt = du.json_load(_mp.encode_subtitle_file)
    encode_qa = du.json_load(_mp.encode_qa_file)
    print(find_max_length(encode_qa, encode_subt))


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--rm', action='store_true', help='Remove pre-processing files.')
    parser.add_argument('--max', action='store_true', help='Find maximal length of all input.')
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.rm:
        remove_all()
    if args.max:
        print_max()
    else:
        qa = QA().include(video_clips=True).get()
        print(len(qa))
        video_data = du.json_load(_mp.video_data_file)
        frame_time = FrameTime().get()
        subtitle = Subtitle().get()
        tokenize_subt = subtitle_process(video_data, frame_time, subtitle)
        tokenize_qa = tokenize_question_answer(qa)

        vocab = create_vocab_embedding(tokenize_subt, tokenize_qa)
        encode_sentence(tokenize_subt, tokenize_qa, vocab)


if __name__ == '__main__':
    main()

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

# from glob import glob
# from os.path import join, isdir

_mp = MovieQAPath()
_ep = EmbeddingPath()


def binary_search(v, t):
    # Find first one larger than t in a sorted array.
    lower = 0
    upper = len(v) - 1
    pivot = (lower + upper) // 2
    while lower < upper:
        if v[pivot] > t:
            upper = pivot
        else:
            lower = pivot + 1
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
            index = binary_search(subt['start'], time)
            temp_tokenize[video].append(
                word_tokenize(subt['lines'][index].lower()))

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
    if not exists(_mp.vocab_file):
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
        du.json_dump(res, _mp.vocab_file)

    else:
        res = du.json_load(_mp.vocab_file)

    return res


def create_vocab_embedding(vocab):
    if not exists(_mp.embedding_file):

        gram_vocab = {k: i for i, k in enumerate(du.json_load(_ep.gram_vocab_file))}
        gram_embed = np.load(_ep.gram_embedding_vec_file)
        vocab_embed = np.zeros((len(vocab) + 1, gram_embed.shape[1]), dtype=np.float32)

        for idx, v in enumerate(tqdm(vocab, desc='Create embedding of vocabulary')):
            v_gram = [c for c in v] + [v[i:i + 3] for i in range(len(v) - 2)] + [v[i:i + 6] for i in range(len(v) - 5)]
            v_gram_code = [gram_vocab[gram] for gram in v_gram if gram in gram_vocab]
            vocab_embed[idx + 1] = np.sum(gram_embed[v_gram_code], axis=0)

        norm = np.linalg.norm(vocab_embed, axis=1, keepdims=True)
        norm = np.select([norm > 0], [norm], default=1.)
        print(norm.shape)
        norm_vocab_embed = vocab_embed / norm
        print(norm_vocab_embed.shape)

        np.save(_mp.embedding_file, norm_vocab_embed)


def encode_sentence(tokenize_subt, tokenize_qa, vocab):
    if not exists(_mp.encode_subtitle_file) or not exists(_mp.encode_qa_file):

        encode_subt, encode_qa = tokenize_subt.copy(), tokenize_qa.copy()
        for imdb in tqdm(encode_subt, desc='Encode subtitle'):
            for v in encode_subt[imdb]:
                encode_subt[imdb][v] = [
                    [vocab[w] for w in sent if w in vocab] for sent in encode_subt[imdb][v]
                ]
        for ins in tqdm(encode_qa, desc='Encode question answer'):
            ins['question'] = [vocab[w] for w in ins['question'] if w in vocab]
            ins['answers'] = [[vocab[w] for w in a if w in vocab] for a in ins['answers']]

        du.json_dump(encode_subt, _mp.encode_subtitle_file)
        du.json_dump(encode_qa, _mp.encode_qa_file)


def remove_all():
    # fu.safe_remove(_mp.tokenize_qa)
    # fu.safe_remove(_mp.tokenize_subt)
    fu.safe_remove(_mp.vocab_file)
    fu.safe_remove(_mp.embedding_file)
    fu.safe_remove(_mp.encode_subtitle_file)
    fu.safe_remove(_mp.encode_qa_file)


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--rm', action='store_true', help='Remove pre-processing files.')

    return parser.parse_args()


def main():
    args = arg_parse()
    if args.rm:
        remove_all()
    qa = QA().include(video_clips=True).get()
    print(len(qa))
    video_data = du.json_load(_mp.video_data_file)
    frame_time = FrameTime().get()
    subtitle = Subtitle().get()
    tokenize_subt = subtitle_process(video_data, frame_time, subtitle)
    tokenize_qa = tokenize_question_answer(qa)

    vocab = create_vocab(tokenize_subt, tokenize_qa)
    create_vocab_embedding(vocab)
    encode_sentence(tokenize_subt, tokenize_qa, vocab)


if __name__ == '__main__':
    main()

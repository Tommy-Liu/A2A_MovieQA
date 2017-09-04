import re
import os
import json
import imageio

import numpy as np

from glob import glob
from tqdm import tqdm
from os.path import join
from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize
from video_preprocessing import get_base_name, get_base_name_without_ext, clean_token

video_img = './video_img'
UNK = 'UNK'
IMAGE_PATTERN_ = '*.jpg'

qa_file_name = './avail_preprocessing_qa.json'
sep_vocab_file_name = './avail_separate_vocab.json'
all_vocab_file_name = './avail_all_vocab.json'
info_file = './info.json'


def is_in(a, b):
    '''
    Is a a subset of b ?
    :param a: set a
    :param b: set b
    :return: True or False
    '''
    return set(a).issubset(set(b))


def get_imdb_key(d):
    '''
    Get imdb key form the directory.
    :param d: the directory.
    :return: key of imdb.
    '''
    return re.search(r'(tt.*?)(?=\.)', d).group(0)


def qid_split(qa_):
    return qa_['qid'].split(':')[0]


def insert_unk(vocab, inverse_vocab):
    vocab[UNK] = len(vocab)
    inverse_vocab.append(UNK)
    return vocab, inverse_vocab


def build_vocab(counter):
    sorted_vocab = sorted(counter.items(),
                          key=lambda t: t[1],
                          reverse=True)
    vocab = {
        item[0]: idx
        for idx, item in enumerate(sorted_vocab)
    }
    inverse_vocab = [key for key in vocab.keys()]
    return insert_unk(vocab, inverse_vocab)


def cleans(subtitles):
    for key in subtitles.keys():
        subtitles[key] = [
            clean_token(sent)
            for sent in subtitles[key]
        ]
    return subtitles


def get_split(qa, avail_video_metadata):
    avail_qa_train = []
    total_qa_train = []
    avail_qa_test = []
    total_qa_test = []
    avail_qa_val = []
    total_qa_val = []

    for qa_ in tqdm(qa, desc='Get available split'):
        v_c = [get_base_name_without_ext(vid) for vid in qa_['video_clips']]
        if v_c:
            if is_in(v_c, avail_video_metadata['list']):
                if qid_split(qa_) == 'train':
                    avail_qa_train.append(qa_)
                elif qid_split(qa_) == 'test':
                    avail_qa_test.append(qa_)
                else:
                    avail_qa_val.append(qa_)
            if qid_split(qa_) == 'train':
                total_qa_train.append(qa_)
            elif qid_split(qa_) == 'test':
                total_qa_test.append(qa_)
            else:
                total_qa_val.append(qa_)

    return avail_qa_train, avail_qa_test, avail_qa_val, \
           total_qa_train, total_qa_test, total_qa_val


def tokenize_sentences(qa_list, subtitles, is_train=False):
    if is_train:
        counter_q = Counter()
        counter_a = Counter()
        counter_s = Counter()

    for qa_ in tqdm(qa_list, desc='Tokenize sentences'):
        # Tokenize sentences
        qa_['tokenize_question'] = word_tokenize(qa_['question'])
        qa_['tokenize_answer'] = [word_tokenize(aa) for aa in qa_['answers']]
        qa_['tokenize_video_subtitle'] = [
            subtitles[get_base_name_without_ext(vid)]
            for vid in qa_['video_clips']
        ]
        if is_train:
            # Update counters
            counter_q.update(qa_['tokenize_question'])
            for tokens in qa_['tokenize_answer']:
                counter_a.update(tokens)
            for subt in qa_['tokenize_video_subtitle']:
                for sent in subt:
                    counter_s.update(sent)
    if is_train:
        counter_total = counter_q + counter_a + counter_s
        return counter_q, counter_a, counter_s, counter_total


def encode_sentences(qa_list, vocab_q, vocab_a, vocab_s):
    for qa_ in tqdm(qa_list, desc='Encode sentences'):
        qa_['encoded_answer'] = [
            [vocab_a[word] if word in vocab_a else vocab_a[UNK] for word in aa]
            for aa in qa_['tokenize_answer']
        ]
        qa_['encoded_question'] = [
            vocab_q[word] if word in vocab_q else vocab_q[UNK] for word in qa_['tokenize_question']
        ]
        qa_['encoded_subtitle'] = [
            [
                [vocab_s[word] if word in vocab_s else vocab_s[UNK] for word in sent]
                if sent != [] else [vocab_s[UNK]] for sent in subt
            ]
            for subt in qa_['tokenize_video_subtitle']
        ]
    print(qa_['encoded_subtitle'][0][:10])
    return qa_list


def main():
    avail_video_metadata = json.load(open('avail_video_metadata.json', 'r'))
    avail_video_subtitle = json.load(open('avail_video_subtitle.json', 'r'))
    qa = json.load(open('../MovieQA_benchmark/data/qa.json'))
    print('Loading json file done!!')
    # split = json.load(open('../MovieQA_benchmark/data/splits.json'))
    # unavail_list = [get_base_name(d) for d in avail_video_metadata['unavailable']]

    avail_qa_train, avail_qa_test, avail_qa_val, \
    total_qa_train, total_qa_test, total_qa_val = get_split(qa, avail_video_metadata)

    print('Available qa # : train | test | val ')
    print('                 %5d   %4d   %3d' % (len(avail_qa_train),
                                                len(avail_qa_test),
                                                len(avail_qa_val)))
    print('Total qa # :     train | test | val ')
    print('                 %5d   %4d   %3d' % (len(total_qa_train),
                                                len(total_qa_test),
                                                len(total_qa_val)))

    counter_q, counter_a, counter_s, counter_total = \
        tokenize_sentences(avail_qa_train,
                           avail_video_subtitle,
                           is_train=True)

    # Build vocab
    vocab_q, inverse_vocab_q = build_vocab(counter_q)
    vocab_a, inverse_vocab_a = build_vocab(counter_a)
    vocab_s, inverse_vocab_s = build_vocab(counter_s)
    vocab_total, inverse_vocab_total = build_vocab(counter_total)

    # encode sentences
    tokenize_sentences(avail_qa_test, avail_video_subtitle)
    tokenize_sentences(avail_qa_val, avail_video_subtitle)
    encode_sentences(avail_qa_train, vocab_q, vocab_a, vocab_s)
    encode_sentences(avail_qa_test, vocab_q, vocab_a, vocab_s)
    encode_sentences(avail_qa_val, vocab_q, vocab_a, vocab_s)

    avail_preprocessing_qa = {
        'avail_qa_train': avail_qa_train,
        'avail_qa_test': avail_qa_test,
        'avail_qa_val': avail_qa_val,
    }

    vocab_sep = {
        'vocab_q': vocab_q,
        'vocab_a': vocab_a,
        'vocab_s': vocab_s,
        'inverse_vocab_q': inverse_vocab_q,
        'inverse_vocab_a': inverse_vocab_a,
        'inverse_vocab_s': inverse_vocab_s,
    }

    vocab_all = {
        'vocab_total': vocab_total,
        'inverse_vocab_total': inverse_vocab_total
    }

    os.remove(qa_file_name)
    os.remove(sep_vocab_file_name)
    os.remove(all_vocab_file_name)

    json.dump(avail_preprocessing_qa, open(qa_file_name, 'w'))
    json.dump(vocab_sep, open(sep_vocab_file_name, 'w'))
    json.dump(vocab_all, open(all_vocab_file_name, 'w'))
    # make sure to use new vocab
    os.remove(info_file)


if __name__ == '__main__':
    main()

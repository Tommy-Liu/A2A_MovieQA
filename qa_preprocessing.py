import re
import json
import imageio

import numpy as np

from glob import glob
from tqdm import tqdm
from os.path import join
from collections import Counter
from nltk.tokenize import word_tokenize
from video_preprocessing import get_base_name, get_base_name_without_ext, clean_token


video_img = './video_img'
UNK = 'UNK'
IMAGE_PATTERN_ = '*.jpg'



def is_in(a, b):
    return set(a).issubset(set(b))


def get_imdb_key(d):
    return re.search(r'(tt.*?)(?=\.)', d).group(0)


def qid_split(qa_):
    return qa_['qid'].split(':')[0]


def build_vocab(counter):
    vocab = {
        item[0]: idx
        for idx, item in enumerate(counter.most_common())
    }
    inverse_vocab = [key for key in vocab.keys()]
    return vocab, inverse_vocab


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

    for qa_ in tqdm(qa):
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

    return avail_qa_train,  avail_qa_test, avail_qa_val, \
           total_qa_train, total_qa_test, total_qa_val


def tokenize_and_build_vocab(qa_list, subtitles):
    counter_q = Counter()
    counter_a = Counter()
    counter_s = Counter()
    for qa_ in tqdm(qa_list):
        # Tokenize sentences
        qa_['tokenize_question'] = word_tokenize(qa_['question'])
        qa_['tokenize_answer'] = [word_tokenize(aa) for aa in qa_['answers']]
        qa_['video_img'] = [
            glob(join(video_img, get_base_name_without_ext(vid),
                      IMAGE_PATTERN_))
            for vid in qa_['video_clips']
        ]
        qa_['video_subtitle'] = [
            subtitles[get_base_name_without_ext(vid)]
            for vid in qa_['video_clips']
        ]
        qa_['tokenize_video_subtitle'] = [
            [word_tokenize(sent) for sent in subt]
            for subt in qa_['video_subtitle']
        ]
        # Update counters
        counter_q.update(qa_['tokenize_question'])
        for tokens in qa_['tokenize_answer']:
            counter_a.update(tokens)
        for subt in qa_['tokenize_video_subtitle']:
            for sent in subt:
                counter_s.update(sent)
    # Build vocab
    vocab_q, inverse_vocab_q = build_vocab(counter_q)
    vocab_a, inverse_vocab_a = build_vocab(counter_a)
    vocab_s, inverse_vocab_s = build_vocab(counter_s)
    counter_total = counter_q + counter_a + counter_s

    return counter_q, counter_a, counter_s, \
           vocab_q, inverse_vocab_q,\
           vocab_a, inverse_vocab_a, \
           vocab_s, inverse_vocab_s, counter_total


def encode_sentences(qa_list):
    pass

def main():
    avail_video_metadata = json.load(open('avail_video_metadata.json', 'r'))
    qa = json.load(open('../MovieQA_benchmark/data/qa.json'))
    split = json.load(open('../MovieQA_benchmark/data/splits.json'))
    unavail_list = [get_base_name(d) for d in avail_video_metadata['unavailable']]

    # Fuck those subtitle tokens
    avail_video_metadata['subtitle'] = cleans(avail_video_metadata['subtitle'])

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

    counter_q, counter_a, counter_s, \
    vocab_q, inverse_vocab_q, \
    vocab_a, inverse_vocab_a, \
    vocab_s, inverse_vocab_s, counter_total = tokenize_and_build_vocab(avail_qa_train,
                                                                       avail_video_metadata['subtitle'])




if __name__ == '__main__':
    main()

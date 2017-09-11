import json
import re
from collections import Counter

from nltk.tokenize import word_tokenize
from tqdm import tqdm

from config import MovieQAConfig
from data_utils import clean_token, get_base_name_without_ext, \
    exist_then_remove

config = MovieQAConfig()
video_img = config.video_img
UNK = config.UNK

IMAGE_PATTERN_ = '*.jpg'

qa_file_name = config.qa_file_name
tokenize_file_name = config.tokenize_file_name
encode_file_name = config.encode_file_name
sep_vocab_file_name = config.sep_vocab_file_name
all_vocab_file_name = config.all_vocab_file_name
info_file = config.info_file


def is_in(a, b):
    """
    Is a a subset of b ?
    """
    return set(a).issubset(set(b))


def get_imdb_key(d):
    """
    Get imdb key form the directory.
    :param d: the directory.
    :return: key of imdb.
    """
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
    counter_q = Counter()
    counter_a = Counter()
    counter_s = Counter()
    tokenize_qa_list = []
    for qa_ in tqdm(qa_list, desc='Tokenize sentences'):
        # Tokenize sentences
        tokenize_qa_ = {
            'tokenize_question': word_tokenize(qa_['question']),
            'tokenize_answer': [word_tokenize(aa) for aa in qa_['answers']],
            'tokenize_video_subtitle': [
                subtitles[get_base_name_without_ext(vid)]
                for vid in qa_['video_clips']
            ],
            'video_clips': qa_['video_clips'],
            'correct_index': qa_['correct_index']
        }
        tokenize_qa_list.append(tokenize_qa_)
        if is_train:
            # Update counters
            counter_q.update(tokenize_qa_['tokenize_question'])
            for tokens in tokenize_qa_['tokenize_answer']:
                counter_a.update(tokens)
            for subt in tokenize_qa_['tokenize_video_subtitle']:
                for sent in subt:
                    counter_s.update(sent)
    if is_train:
        counter_total = counter_q + counter_a + counter_s
        return tokenize_qa_list, counter_q, counter_a, counter_s, counter_total
    else:
        return tokenize_qa_list


def encode_sentences(qa_list, vocab_q, vocab_a, vocab_s):
    encode_qa_list = []
    for qa_ in tqdm(qa_list, desc='Encode sentences'):
        encode_qa_ = {
            'encoded_answer': [
                [vocab_a[word] if word in vocab_a else vocab_a[UNK] for word in aa]
                for aa in qa_['tokenize_answer']
            ],
            'encoded_question': [
                vocab_q[word] if word in vocab_q else vocab_q[UNK] for word in qa_['tokenize_question']
            ],
            'encoded_subtitle': [
                [
                    [vocab_s[word] if word in vocab_s else vocab_s[UNK] for word in sent]
                    if sent != [] else [vocab_s[UNK]] for sent in subt
                ]
                for subt in qa_['tokenize_video_subtitle']
            ],
            'video_clips': qa_['video_clips'],
            'correct_index': qa_['correct_index']
        }
        encode_qa_list.append(encode_qa_)
    # print(qa_['encoded_subtitle'][0][:10])
    return encode_qa_list


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

    tokenize_qa_train, counter_q, counter_a, counter_s, counter_total = \
        tokenize_sentences(avail_qa_train,
                           avail_video_subtitle,
                           is_train=True)

    # Build vocab
    vocab_q, inverse_vocab_q = build_vocab(counter_q)
    vocab_a, inverse_vocab_a = build_vocab(counter_a)
    vocab_s, inverse_vocab_s = build_vocab(counter_s)
    vocab_total, inverse_vocab_total = build_vocab(counter_total)

    # encode sentences
    tokenize_qa_test = tokenize_sentences(avail_qa_test, avail_video_subtitle)
    tokenize_qa_val = tokenize_sentences(avail_qa_val, avail_video_subtitle)
    encode_qa_train = encode_sentences(tokenize_qa_train, vocab_q, vocab_a, vocab_s)
    encode_qa_test = encode_sentences(tokenize_qa_test, vocab_q, vocab_a, vocab_s)
    encode_qa_val = encode_sentences(tokenize_qa_val, vocab_q, vocab_a, vocab_s)

    avail_preprocessing_qa = {
        'avail_qa_train': avail_qa_train,
        'avail_qa_test': avail_qa_test,
        'avail_qa_val': avail_qa_val,
    }

    tokenize_qa = {
        'tokenize_qa_train': tokenize_qa_train,
        'tokenize_qa_test': tokenize_qa_test,
        'tokenize_qa_val': tokenize_qa_val,
    }

    encode_qa = {
        'encode_qa_train': encode_qa_train,
        'encode_qa_test': encode_qa_test,
        'encode_qa_val': encode_qa_val,
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

    exist_then_remove(qa_file_name)
    exist_then_remove(tokenize_file_name)
    exist_then_remove(encode_file_name)
    exist_then_remove(sep_vocab_file_name)
    exist_then_remove(all_vocab_file_name)

    json.dump(avail_preprocessing_qa, open(qa_file_name, 'w'))
    json.dump(tokenize_qa, open(tokenize_file_name, 'w'))
    json.dump(encode_qa, open(encode_file_name, 'w'))
    json.dump(vocab_sep, open(sep_vocab_file_name, 'w'))
    json.dump(vocab_all, open(all_vocab_file_name, 'w'))
    # make sure to use new vocab
    exist_then_remove(info_file)


if __name__ == '__main__':
    main()

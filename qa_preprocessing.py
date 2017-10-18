import argparse
import re
import time
import ujson as json
from collections import Counter
from functools import wraps

import numpy as np
# from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize import word_tokenize  # , RegexpTokenizer, TweetTokenizer
from tqdm import tqdm, trange

import data_utils as du
from config import MovieQAConfig

config = MovieQAConfig()
video_img = config.video_img_dir
UNK = config.UNK

IMAGE_PATTERN_ = '*.jpg'

total_qa_file_name = config.total_split_qa_file
tokenize_file_name = config.avail_tokenize_qa_file
encode_file_name = config.avail_encode_qa_file
all_vocab_file_name = config.all_vocab_file

tokenize_func = word_tokenize


# tokenizer = RegexpTokenizer("[\w']+")
# tokenizer = TweetTokenizer()
# tokenizer = MosesTokenizer()
# tokenize_func = partial(tokenizer.tokenize, escape=False)


def bar(func, ch='='):
    @wraps(func)
    def wrapper(s):
        print(ch * (max([len(e) for e in s]) + 5))
        func(s)
        print(ch * (max([len(e) for e in s]) + 5))

    return wrapper


@bar
def pprint(s):
    print('\n'.join(s))


def get_imdb_key(d):
    """
    Get imdb key form the directory.
    :param d: the directory.
    :return: key of imdb.
    """
    return re.search(r'(tt.*?)(?=\.)', d).group(0)


def qid_split(qa_):
    return qa_['qid'].split(':')[0]


def load_embedding(file):
    embedding = {}

    with open(file, 'r') as f:
        num, dim = [int(comp) for comp in f.readline().strip().split()]
        for _ in trange(num, desc='Load word embedding %dd' % dim):
            comp = f.readline().strip().split()
            word, vec = comp[:-dim], comp[-dim:]
            word = ' '.join(word)
            vec = [float(e) for e in vec]
            embedding[word] = vec

    return embedding


def insert_unk(qa_embedding, vocab, inverse_vocab):
    vocab[UNK] = len(vocab)
    inverse_vocab.append(UNK)
    qa_embedding[UNK] = np.mean(list(qa_embedding.values()), axis=0).tolist()
    return qa_embedding, vocab, inverse_vocab


def build_vocab(counter, embedding):
    qa_embedding = {}
    sorted_counter = counter.most_common()
    # sorted_counter = sorted(counter.items(),
    #                         key=lambda t: t[1],
    #                         reverse=True)
    for idx, item in tqdm(enumerate(sorted_counter), desc='Build vocab:'):
        if item[0] in embedding.keys() and item[1] > config.vocab_thr:
            qa_embedding[item[0]] = embedding[item[0]]

    pprint([
        'Original vocabulary size: %d' % len(counter),
        'Embedding vocabulary size: %d' % len(qa_embedding),
        'Embedding vocabulary coverage: %.2f %%' % (len(qa_embedding) / len(counter) * 100),
    ])
    vocab = {k: i for i, k in enumerate(qa_embedding.keys())}
    inverse_vocab = [k for k in qa_embedding.keys()]
    return insert_unk(qa_embedding, vocab, inverse_vocab)


def legacy_build_vocab(counter):
    sorted_counter = sorted(counter.items(),
                            key=lambda t: t[1],
                            reverse=True)
    vocab = {
        item[0]: idx
        for idx, item in enumerate(sorted_counter)
        if item[1] > config.vocab_thr
    }
    inverse_vocab = [key for key in vocab.keys()]
    return insert_unk(vocab, inverse_vocab)


def get_split(qa, video_data):
    total_qa = {
        'train': [],
        'test': [],
        'val': [],
    }
    for qa_ in tqdm(qa, desc='Get available split'):
        total_qa[qid_split(qa_)].append({
            "qid": qa_['qid'],
            "question": qa_['question'],
            "answers": qa_['answers'],
            "imdb_key": qa_['imdb_key'],
            "correct_index": qa_['correct_index'],
            "mv+sub": qa_['video_clips'] != [],
            "video_clips": [du.get_base_name_without_ext(vid)
                            for vid in qa_['video_clips'] if video_data[du.get_base_name_without_ext(vid)]['avail']],
        })
        total_qa[qid_split(qa_)][-1]['avail'] = (total_qa[qid_split(qa_)][-1]['video_clips'] != [])
    return total_qa


def tokenize_sentences(qa_list, subtitles, is_train=False):
    vocab_counter = Counter()
    tokenize_qa_list = []

    for qa_ in tqdm(qa_list, desc='Tokenize sentences'):
        # Tokenize sentences
        if qa_['avail']:
            tokenize_qa_list.append(
                {
                    'tokenize_question': tokenize_func(qa_['question'].lower().strip()),
                    'tokenize_answer': [tokenize_func(aa.lower().strip())
                                        for aa in qa_['answers']],
                    'video_clips': qa_['video_clips'],
                    'correct_index': qa_['correct_index']
                }
            )
            if is_train:
                # Update counters
                vocab_counter.update(tokenize_qa_list[-1]['tokenize_question'])

                for ans in tokenize_qa_list[-1]['tokenize_answer']:
                    vocab_counter.update(ans)

                for v in qa_['video_clips']:
                    for sub in subtitles[v]:
                        vocab_counter.update(sub)

    if is_train:
        return tokenize_qa_list, vocab_counter
    else:
        return tokenize_qa_list


def encode_subtitles(subtitles, vocab):
    encode_sub = {}
    for key in subtitles.keys():
        if subtitles[key]:
            encode_sub[key] = {
                'subtitle': [
                    [vocab[word] if word in vocab else vocab[UNK] for word in sub]
                    if sub != [] else [vocab[UNK]]
                    for sub in subtitles[key]['subtitle']
                ],
                'subtitle_index': subtitles[key]['subtitle_index'],
                'frame_time': subtitles[key]['frame_time'],
            }
        else:
            encode_sub[key] = {}
    return encode_sub


def encode_sentences(qa_list, vocab):
    encode_qa_list = []
    for qa_ in tqdm(qa_list, desc='Encode sentences'):
        encode_qa_list.append({
            'encoded_answer': [
                [vocab[word] if word in vocab else vocab[UNK] for word in aa]
                for aa in qa_['tokenize_answer']
            ],
            'encoded_question': [
                vocab[word] if word in vocab else vocab[UNK] for word in qa_['tokenize_question']
            ],
            'video_clips': qa_['video_clips'],
            'correct_index': qa_['correct_index']
        })

    # print(qa_['encoded_subtitle'][0][:10])
    return encode_qa_list


def main():
    start_time = time.time()
    video_data = du.load_json(config.video_data_file)
    video_subtitle = du.load_json(config.subtitle_file)
    qa = json.load(open(config.qa_file, 'r'))
    embed_file = None
    if args.embedding == 'word2vec':
        embed_file = config.word2vec_file
    elif args.embedding == 'fasttext':
        embed_file = config.fasttext_file
    elif args.embedding == 'glove':
        embed_file = config.glove_file
    embedding = None
    if embed_file:
        embedding = load_embedding(embed_file)
    print('Loading json file done!! Take %.4f sec.' % (time.time() - start_time))

    total_qa = get_split(qa, video_data)

    print('Available qa # : train | test | val ')
    print('                 %5d   %4d   %3d' % (len([0 for qa_ in total_qa['train'] if qa_['avail']]),
                                                len([0 for qa_ in total_qa['test'] if qa_['avail']]),
                                                len([0 for qa_ in total_qa['val'] if qa_['avail']])))
    print('Mv+Sub qa # :    train | test | val ')
    print('                 %5d   %4d   %3d' % (len([0 for qa_ in total_qa['train'] if qa_['mv+sub']]),
                                                len([0 for qa_ in total_qa['test'] if qa_['mv+sub']]),
                                                len([0 for qa_ in total_qa['val'] if qa_['mv+sub']])))
    print('Total qa # :     train | test | val ')
    print('                 %5d   %4d   %3d' % (len(total_qa['train']),
                                                len(total_qa['test']),
                                                len(total_qa['val'])))

    tokenize_qa_train, unavail_word, vocab_counter = \
        tokenize_sentences(total_qa['train'],
                           video_subtitle,
                           is_train=True)
    # json.dump(unavail_word_to_subtitle, open('unavail_word_to_subtitle.json', 'w'), indent=4)
    # Build vocab
    qa_embedding, vocab, inverse_vocab = build_vocab(vocab_counter, embedding)

    # encode sentences
    tokenize_qa_test = tokenize_sentences(total_qa['test'], video_subtitle)
    tokenize_qa_val = tokenize_sentences(total_qa['val'], video_subtitle)
    encode_sub = encode_subtitles(video_subtitle, vocab)
    encode_qa_train = encode_sentences(tokenize_qa_train, vocab)
    encode_qa_test = encode_sentences(tokenize_qa_test, vocab)
    encode_qa_val = encode_sentences(tokenize_qa_val, vocab)

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
    vocab_all = {
        'vocab': vocab,
        'inverse_vocab': inverse_vocab,
    }

    du.exist_then_remove(total_qa_file_name)
    du.exist_then_remove(tokenize_file_name)
    du.exist_then_remove(encode_file_name)
    du.exist_then_remove(all_vocab_file_name)
    du.exist_then_remove(config.encode_subtitle_file)

    du.write_json(total_qa, total_qa_file_name)
    du.write_json(tokenize_qa, tokenize_file_name)
    du.write_json(encode_qa, encode_file_name)
    du.write_json(vocab_all, all_vocab_file_name)
    du.write_json(encode_sub, config.encode_subtitle_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='word2vec', help='The embedding method we want to use.')
    args = parser.parse_args()
    main()

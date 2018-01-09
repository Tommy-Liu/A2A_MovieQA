import argparse
import re
import time
import ujson as json
from collections import Counter
from os.path import exists

import numpy as np
# from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize import word_tokenize  # , RegexpTokenizer, TweetTokenizer
from tqdm import tqdm

from config import MovieQAConfig
from utils import data_utils as du
from utils import func_utils as fu

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



def get_imdb_key(d):
    """
    Get imdb key form the directory.
    :param d: the directory.
    :return: key of imdb.
    """
    return re.search(r'(tt.*?)(?=\.)', d).group(0)


def qid_split(qa_):
    return qa_['qid'].split(':')[0]


def insert_unk(qa_embedding, vocab, inverse_vocab):
    vocab[UNK] = len(vocab)
    inverse_vocab.append(UNK)
    qa_embedding[UNK] = np.mean(list(qa_embedding.values()), axis=0).tolist()
    return qa_embedding, vocab, inverse_vocab


def build_vocab(counter, embedding=None):
    qa_embedding = {}
    sorted_counter = counter.most_common()
    # sorted_counter = sorted(counter.items(),
    #                         key=lambda t: t[1],
    #                         reverse=True)
    for idx, item in tqdm(enumerate(sorted_counter), desc='Build vocab:'):
        if item[1] > config.vocab_thr:
            if embedding and item[0] in embedding.keys():
                qa_embedding[item[0]] = embedding[item[0]]
            else:
                qa_embedding[item[0]] = item[1]

    fu.block_print([
        'Original vocabulary size: %d' % len(counter),
        'Embedding vocabulary size: %d' % len(qa_embedding),
        'Embedding vocabulary coverage: %.2f %%' % (len(qa_embedding) / len(counter) * 100),
    ])
    vocab = {k: i for i, k in enumerate(qa_embedding.keys())}
    inverse_vocab = list(qa_embedding.keys())
    return insert_unk(qa_embedding, vocab, inverse_vocab)


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
            "video_clips": [fu.get_base_name_without_ext(vid)
                            for vid in qa_['video_clips'] if video_data[fu.get_base_name_without_ext(vid)]['avail']],
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

    return tokenize_qa_list, vocab_counter


def encode_subtitles(subtitles, vocab, shot_boundary, subtitle_shot):
    encode_sub = {}
    for key in subtitles.keys():
        if subtitles[key]:
            encode_sub[key] = {
                'subtitle': [
                    [vocab[word] if word in vocab else vocab[UNK] for word in sub]
                    if sub != [] else [vocab[UNK]]
                    for sub in subtitles[key]
                ],
                'subtitle_shot': subtitle_shot[key],
                'shot_boundary': shot_boundary[key]
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
    video_data = du.json_load(config.video_data_file)
    video_subtitle = du.json_load(config.subtitle_file)
    shot_boundary = du.json_load(config.shot_boundary_file)
    subtitle_shot = du.json_load(config.subtitle_shot_file)

    qa = json.load(open(config.qa_file, 'r'))

    embed_file = None
    avail_embed_file = None
    avail_embed_npy_file = None
    if args.embedding == 'word2vec':
        embed_file = config.word2vec_file
        avail_embed_file = config.w2v_embedding_file
        avail_embed_npy_file = config.w2v_embedding_npy_file
    elif args.embedding == 'fasttext':
        embed_file = config.fasttext_file
        avail_embed_file = config.ft_embedding_file
        avail_embed_npy_file = config.ft_embedding_npy_file
    elif args.embedding == 'glove':
        embed_file = config.glove_file
        avail_embed_file = config.glove_embedding_file
        avail_embed_npy_file = config.glove_embedding_npy_file

    embedding = None
    embed_exist = exists(avail_embed_file) and exists(avail_embed_npy_file)
    if embed_file and not embed_exist:
        embedding = du.load_w2v(embed_file)

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

    tokenize_qa_train, vocab_counter = \
        tokenize_sentences(total_qa['train'],
                           video_subtitle,
                           is_train=not embed_exist)
    # Build vocab
    if embed_file:
        if not embed_exist:
            qa_embedding, vocab, inverse_vocab = build_vocab(vocab_counter, embedding)
            fu.exist_then_remove(avail_embed_file)
            du.json_dump(inverse_vocab, avail_embed_file)
            fu.exist_then_remove(avail_embed_npy_file)
            np.save(avail_embed_npy_file,
                    np.array([e for e in qa_embedding.values()], dtype=np.float32))
        else:
            inverse_vocab = du.json_load(avail_embed_file)
            vocab = {k: i for i, k in enumerate(inverse_vocab)}
    else:
        _, vocab, inverse_vocab = build_vocab(vocab_counter)

    # encode sentences
    tokenize_qa_test, _ = tokenize_sentences(total_qa['test'], video_subtitle)
    tokenize_qa_val, _ = tokenize_sentences(total_qa['val'], video_subtitle)
    encode_sub = encode_subtitles(video_subtitle, vocab, shot_boundary, subtitle_shot)
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

    fu.exist_then_remove(total_qa_file_name)
    fu.exist_then_remove(tokenize_file_name)
    fu.exist_then_remove(encode_file_name)
    fu.exist_then_remove(all_vocab_file_name)
    fu.exist_then_remove(config.encode_subtitle_file)

    du.json_dump(total_qa, total_qa_file_name)
    du.json_dump(tokenize_qa, tokenize_file_name)
    du.json_dump(encode_qa, encode_file_name)
    du.json_dump(vocab_all, all_vocab_file_name)
    du.json_dump(encode_sub, config.encode_subtitle_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', default='word2vec', help='The embedding method we want to use.')
    args = parser.parse_args()
    main()

import re
import time
import ujson as json
from collections import Counter

from nltk.tokenize import word_tokenize
from tqdm import tqdm

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


def tokenize_sentences(qa_list, is_train=False):
    vocab_counter = Counter()
    tokenize_qa_list = []
    for qa_ in tqdm(qa_list, desc='Tokenize sentences'):
        # Tokenize sentences
        if qa_['avail']:
            tokenize_qa_list.append(
                {
                    'tokenize_question': word_tokenize(qa_['question']),
                    'tokenize_answer': [word_tokenize(aa) for aa in qa_['answers']],
                    'video_clips': qa_['video_clips'],
                    'correct_index': qa_['correct_index']
                }
            )
            if is_train:
                # Update counters
                vocab_counter.update(tokenize_qa_list[-1]['tokenize_question'])
                for ans in tokenize_qa_list[-1]['tokenize_answer']:
                    vocab_counter.update(ans)

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
    video_data = json.load(open(config.video_data_file, 'r'))
    video_subtitle = json.load(open(config.subtitle_file, 'r'))
    qa = json.load(open(config.qa_file))
    print('Loading json file done!! Take %.4f sec.' % (time.time() - start_time))
    # split = json.load(open('../MovieQA_benchmark/data/splits.json'))
    # unavail_list = [get_base_name(d) for d in avail_video_metadata['unavailable']]

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

    tokenize_qa_train, vocab_counter = tokenize_sentences(total_qa['train'],
                                                          is_train=True)
    for key in video_subtitle.keys():
        if video_subtitle[key]:
            for sub in video_subtitle[key]['subtitle']:
                vocab_counter.update(sub)
    # Build vocab
    vocab, inverse_vocab = build_vocab(vocab_counter)

    # encode sentences
    tokenize_qa_test = tokenize_sentences(total_qa['test'])
    tokenize_qa_val = tokenize_sentences(total_qa['val'])
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
    main()

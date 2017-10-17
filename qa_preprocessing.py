import os
import re
import time
import ujson as json
from collections import Counter
from functools import partial

from nltk.tokenize.moses import MosesTokenizer
# from nltk.tokenize import word_tokenize, RegexpTokenizer, TweetTokenizer
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

# tokenize_func = word_tokenize
# tokenizer = RegexpTokenizer("[\w']+")
# tokenizer = TweetTokenizer()
tokenizer = MosesTokenizer()
tokenize_func = partial(tokenizer.tokenize, escape=False)


def get_imdb_key(d):
    """
    Get imdb key form the directory.
    :param d: the directory.
    :return: key of imdb.
    """
    return re.search(r'(tt.*?)(?=\.)', d).group(0)


def qid_split(qa_):
    return qa_['qid'].split(':')[0]


def load_embedding():
    if os.path.exists(config.fasttext_vocb_file):
        with open(config.fasttext_vocb_file, 'r') as f:
            embedding = json.load(f)
    else:
        embedding = {}
        with open(config.fasttext_file, 'r') as f:
            num, dim = [int(comp) for comp in f.readline().strip().split()]
            for _ in trange(num, desc='Load word embedding:'):
                word, *vec = f.readline().strip().split()
                vec = [float(e) for e in vec]
                embedding[word] = vec
        with open(config.fasttext_vocb_file, 'w') as f:
            json.dump(embedding, f, indent=4)
    return embedding


def nn():
    pass


def insert_unk(vocab, inverse_vocab):
    vocab[UNK] = len(vocab)
    inverse_vocab.append(UNK)
    return vocab, inverse_vocab


def build_vocab(counter, embedding):
    qa_embedding = {}
    sorted_counter = sorted(counter.items(),
                            key=lambda t: t[1],
                            reverse=True)
    for idx, item in tqdm(enumerate(sorted_counter), desc='Build vocab:'):
        if item[0] in embedding.keys() and item[1] > config.vocab_thr:
            qa_embedding[item[0]] = embedding[item[0]]
    print('Fasttext vocabulary coverage: %.2f %%' % (len(qa_embedding) / len(counter) * 100))


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


def tokenize_sentences(qa_list, embedding, unavail_word_to_subtitle, is_train=False):
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
                for w in tokenize_qa_list[-1]['tokenize_question']:
                    if w not in embedding.keys() and \
                                    ' '.join(tokenize_qa_list[-1]['tokenize_question']) \
                                    not in unavail_word_to_subtitle.setdefault(w, []):
                        unavail_word_to_subtitle[w] \
                            .append(' '.join(tokenize_qa_list[-1]['tokenize_question']))
                for ans in tokenize_qa_list[-1]['tokenize_answer']:
                    vocab_counter.update(ans)
                    for w in ans:
                        if w not in embedding.keys() and \
                                        ' '.join(ans) not in unavail_word_to_subtitle.setdefault(w, []):
                            unavail_word_to_subtitle[w].append(' '.join(ans))

    if is_train:
        return tokenize_qa_list, unavail_word_to_subtitle, vocab_counter
    else:
        return tokenize_qa_list, unavail_word_to_subtitle


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
    video_subtitle_index = du.load_json(config.subtitle_index_file)
    qa = json.load(open(config.qa_file, 'r'))
    embedding = load_embedding()
    unavail_word_to_subtitle = {}
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

    tokenize_qa_train, unavail_word_to_subtitle, vocab_counter = \
        tokenize_sentences(total_qa['train'],
                           embedding,
                           unavail_word_to_subtitle,
                           is_train=True)
    for key in video_subtitle.keys():
        if video_subtitle[key]:
            for sub in video_subtitle[key]['subtitle']:
                vocab_counter.update(sub)
                for w in sub:
                    if w not in embedding.keys() and \
                                    ' '.join(sub) not in unavail_word_to_subtitle.setdefault(w, []):
                        unavail_word_to_subtitle[w].append(' '.join(sub))
    # json.dump(unavail_word_to_subtitle, open('unavail_word_to_subtitle.json', 'w'), indent=4)
    # Build vocab
    build_vocab(vocab_counter, embedding)
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

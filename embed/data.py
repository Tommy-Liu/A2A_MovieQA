import io
import math
import time
from collections import Counter
from os.path import exists
from pprint import PrettyPrinter

import numpy as np
from tqdm import tqdm, trange

from .args import CommonParameter
from ..utils import data_utils as du
from ..utils import func_utils as fu

pp = PrettyPrinter(indent=2, compact=True)

cp = CommonParameter()
UNK = 'UNK'


def load_embedding_vec(target, embedding_size=cp.embedding_size):
    start_time = time.time()
    if target == 'glove':
        key_file = cp.glove_embedding_key_file
        vec_file = cp.glove_embedding_vec_file
        raw_file = cp.glove_file
        load_fn = load_glove
    elif target == 'w2v':
        key_file = cp.w2v_embedding_key_file
        vec_file = cp.w2v_embedding_vec_file
        raw_file = cp.word2vec_file
        load_fn = load_w2v
    elif target == 'fasttext':
        key_file = cp.ft_embedding_key_file
        vec_file = cp.ft_embedding_vec_file
        raw_file = cp.fasttext_file
        load_fn = load_glove
    else:
        key_file = None
        vec_file = None
        raw_file = None
        load_fn = None

    if exists(key_file) and exists(vec_file):
        embedding_keys = du.json_load(key_file)
        embedding_vecs = np.load(vec_file)
    else:
        embedding = load_fn(raw_file)
        embedding_keys = []
        embedding_vecs = np.zeros((len(embedding), embedding_size), dtype=np.float32)
        for i, k in enumerate(embedding.keys()):
            embedding_keys.append(k)
            embedding_vecs[i] = embedding[k]
        du.json_dump(embedding_keys, key_file)
        np.save(vec_file, embedding_vecs)

    print('Loading embedding done. %.3f s' % (time.time() - start_time))
    return embedding_keys, embedding_vecs


def load_w2v(file):
    embedding = {}

    with open(file, 'r') as f:
        num, dim = [int(comp) for comp in f.readline().strip().split()]
        for _ in trange(num, desc='Load word embedding %dd' % dim):
            word, *vec = f.readline().rstrip().rsplit(sep=' ', maxsplit=dim)
            vec = [float(e) for e in vec]
            embedding[word] = vec
        assert len(embedding) == num, 'Wrong size of embedding.'
    return embedding


def load_glove(filename, embedding_size=cp.embedding_size):
    embedding = {}

    # Read in the data.
    with io.open(filename, 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(tqdm(savefile)):
            tokens = line.rstrip().split(sep=' ', maxsplit=embedding_size)

            word, *entries = tokens

            embedding[word] = [float(x) for x in entries]
            assert len(embedding[word]) == embedding_size, 'Wrong embedding dim.'

    return embedding


def filter_stat(embedding_keys, embedding_vecs, max_length):
    # Filter out non-ascii words
    count, mean, keys, std = 0, 0, {}, 0
    for i, k in enumerate(tqdm(embedding_keys, desc='Filtering...')):
        try:
            k.encode('ascii')
        except UnicodeEncodeError:
            pass
        else:
            count += 1
            kk = k.lower().strip()
            d1 = (len(kk) - mean)
            mean += d1 / count
            d2 = (len(kk) - mean)
            std += d1 * d2
            if len(kk) <= max_length:
                if keys.get(kk, None):
                    if k.strip().islower():
                        keys[k.strip()] = i
                else:
                    keys[k.lower().strip()] = i
    std = math.sqrt(std / count)
    vecs = embedding_vecs[list(keys.values())]

    embedding_keys, embedding_vecs = list(keys.keys()), vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    fu.block_print(['Filtered number of embedding: %d' % len(embedding_keys),
                    'Filtered shape of embedding vec: ' + str(embedding_vecs.shape),
                    'Length\'s mean of keys: %.3f' % mean,
                    'Length\'s std of keys: %.3f' % std,
                    'Mean of embedding vecs: %.6f' % np.mean(np.mean(embedding_vecs, 1)),
                    'Std of embedding vecs: %.6f' % np.std(embedding_vecs),
                    'Mean length of embedding vecs: %.6f' % np.mean(np.linalg.norm(embedding_vecs, axis=1)),
                    'Std length of embedding vecs: %.6f' % np.std(np.linalg.norm(embedding_vecs, axis=1)),
                    ])
    print('Element mean of embedding vec:')
    pp.pprint(np.mean(embedding_vecs, axis=0))
    return embedding_keys, embedding_vecs


def process():
    # tokenize_qa = du.json_load(cp.avail_tokenize_qa_file)
    # subtitle = du.json_load(cp.subtitle_file)
    embedding_keys, embedding_vector = load_embedding_vec(cp.target)

    fu.block_print(['%s\'s # of embedding: %d' % (cp.target, len(embedding_keys)),
                    '%s\'s shape of embedding vec: %s' % (cp.target, str(embedding_vector.shape))])

    embedding_keys, embedding_vector = filter_stat(embedding_keys, embedding_vector, cp.max_length)

    frequency = Counter()
    probability = {}
    embed_char_counter = Counter()
    for k in tqdm(embedding_keys):
        frequency.update([k[:(i + 1)] for i in range(len(k))])
        embed_char_counter.update(k)

    # Calculate the distribution of length 1
    target = [k for k in frequency.keys() if len(k) == 1]
    total = np.sum([frequency[t] for t in target])
    probability.update({t: frequency[t] / total for t in target})

    # Calculate length > 1
    for l in range(2, cp.max_length + 1):
        target = [k for k in frequency.keys() if len(k) == l]
        total = np.sum([frequency[t] for t in target])

        probability.update({t: frequency[t] / total for t in target})

    # traverse(root)
    # print(root)
    if not cp.debug:
        # qa_char_counter = Counter()
        # for k in tokenize_qa.keys():
        #     for qa in tqdm(tokenize_qa[k], desc='Char counting %s' % k):
        #         for w in qa['tokenize_question']:
        #             qa_char_counter.update(w)
        #         for a in qa['tokenize_answer']:
        #             for w in a:
        #                 qa_char_counter.update(w)
        #         for v in qa['video_clips']:
        #             for l in subtitle[v]:
        #                 for w in l:
        #                     qa_char_counter.update(w)

        du.json_dump(embed_char_counter, cp.embed_char_counter_file)
        # du.json_dump(qa_char_counter, cp.qa_char_counter_file)

        # count_array = np.array(list(embed_char_counter.values()), dtype=np.float32)
        # m, v, md, f = np.mean(count_array), np.std(count_array), np.median(count_array), np.percentile(count_array, 95)
        # print(m, v, md, f)
        #
        # above_mean = dict(filter(lambda item: item[1] > f, embed_char_counter.items()))
        # below_mean = dict(filter(lambda item: item[1] < f, embed_char_counter.items()))
        # below_occur = set(filter(lambda k: k in qa_char_counter, below_mean.keys()))
        # final_set = below_occur.union(set(above_mean.keys()))
        # du.json_dump(list(final_set) + [UNK], cp.char_vocab_file)
        vocab = list(embed_char_counter.keys()) + [UNK]
        print('Filtered vocab:', vocab)
        du.json_dump(vocab, cp.char_vocab_file)
        # vocab = du.json_load(cp.char_vocab_file)
        encode_embedding_keys = np.ones((len(embedding_keys), cp.max_length), dtype=np.int64) * (len(vocab) - 1)
        length = np.zeros(len(embedding_keys), dtype=np.int64)
        for i, k in enumerate(tqdm(embedding_keys, desc='Encoding...')):
            encode_embedding_keys[i, :len(k)] = [
                vocab.index(ch) if ch in vocab else vocab.index(UNK)
                for ch in k
            ]
            assert all([idx < len(vocab) for idx in encode_embedding_keys[i]]), \
                "Wrong index!"
            length[i] = len(k)
        fu.block_print(['Shape of encoded key: %s' % str(encode_embedding_keys.shape),
                        'Shape of encoded key length: %s' % str(length.shape)])
        start_time = time.time()
        fu.exist_then_remove(cp.encode_embedding_key_file)
        fu.exist_then_remove(cp.encode_embedding_len_file)
        fu.exist_then_remove(cp.encode_embedding_vec_file)
        np.save(cp.encode_embedding_key_file, encode_embedding_keys)
        np.save(cp.encode_embedding_len_file, length)
        np.save(cp.encode_embedding_vec_file, embedding_vector)
        print('Saveing processed data with %.3f s' % (time.time() - start_time))


if __name__ == '__main__':
    process()

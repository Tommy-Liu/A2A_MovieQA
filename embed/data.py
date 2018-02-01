import io
import math
import time
from argparse import ArgumentParser
from collections import Counter
from os.path import exists
from pprint import pprint

import numpy as np
from tqdm import tqdm, trange

from embed.args import EmbeddingPath
from utils import data_utils as du
from utils import func_utils as fu

cp = EmbeddingPath()
UNK = 'UNK'


def load_embedding_vec(target, embedding_size=cp.embedding_size):
    start_time = time.time()
    # File and function setup for embedding
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

    # Check if there already exists pre-loaded file
    if exists(key_file) and exists(vec_file):
        embedding_keys = du.json_load(key_file)
        embedding_vecs = np.load(vec_file)
    else:
        # If not, load it from text file
        embedding_keys, embedding_vecs = load_fn(raw_file)
        du.json_dump(embedding_keys, key_file)
        np.save(vec_file, embedding_vecs)

    print('Loading embedding done. %.3f s' % (time.time() - start_time))
    return embedding_keys, embedding_vecs


def load_w2v(filename):
    embedding_key, embedding_vec = [], []

    with open(filename, 'r') as f:
        # First line is size of vocabulary and dimension of embedding
        num, dim = [int(comp) for comp in f.readline().strip().split()]
        # Split each line of the rest to word and embedding vector
        for _ in trange(num, desc='Load word embedding %dd' % dim):
            word, *vec = f.readline().rstrip().rsplit(sep=' ', maxsplit=dim)
            embedding_key.append(word)
            embedding_vec.append([float(e) for e in vec])
        assert len(embedding_vec[-1]) == num, 'Wrong size of embedding.'
        # Wrap it to a numpy array
        embedding_vec = np.array(embedding_vec, dtype=np.float32)
    return embedding_key, embedding_vec


def load_glove(filename, embedding_size=cp.embedding_size):
    embedding_key, embedding_vec = [], []

    # Read in the data.
    with io.open(filename, 'r', encoding='utf-8') as savefile:
        # Split each line of the rest to word and embedding vector
        for line in tqdm(savefile):
            tokens = line.rstrip().split(sep=' ', maxsplit=embedding_size)

            word, *entries = tokens
            embedding_key.append(word)
            embedding_vec.append([float(x) for x in entries])
            assert len(embedding_vec[-1]) == embedding_size, 'Wrong embedding dim.'
        # Wrap it to a numpy array
        embedding_vec = np.array(embedding_vec, dtype=np.float32)
    return embedding_key, embedding_vec


def filter_stat(embedding_keys, embedding_vector, max_length=0, print_stat=True,
                normalize=True):
    # Filter out non-ascii words and words with '<' and '>'.
    count, mean, keys, std = 0, 0, {}, 0
    for idx, k in enumerate(tqdm(embedding_keys, desc='Filtering')):
        try:
            k.encode('ascii')
        except UnicodeEncodeError:
            pass
        else:
            count += 1
            kk = '<' + k.lower().strip() + '>'
            d1 = len(kk) - mean
            mean += d1 / count
            d2 = len(kk) - mean
            std += d1 * d2
            # Threshold for maximum length, if max_length set
            length_flag = len(kk) <= max_length or not max_length
            lt_gt_flag = not {'<', '>'} & set(k)
            lower_none_flag = keys.get(kk, None) and k.strip().islower() or not keys.get(kk, None)
            if length_flag and lt_gt_flag and lower_none_flag:
                keys[kk] = idx

    std = math.sqrt(std / count)
    vector = np.zeros((len(keys), 300), dtype=np.float32)
    embedding_keys = []
    for idx, k in enumerate(keys):
        embedding_keys.append(k)
        vector[idx] = embedding_vector[keys[k]]

    # Normalize each vector to unit vector
    if normalize:
        embedding_vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
    else:
        embedding_vector = vector

    if print_stat:
        fu.block_print(['Filtered number of embedding: %d' % len(embedding_keys),
                        'Filtered shape of embedding vec: ' + str(embedding_vector.shape),
                        'Length\'s mean of keys: %.3f' % mean,
                        'Length\'s std of keys: %.3f' % std,
                        'Mean of embedding vecs: %.6f' % np.mean(np.mean(embedding_vector, 1)),
                        'Std of embedding vecs: %.6f' % np.std(embedding_vector),
                        'Mean length of embedding vecs: %.6f' % np.mean(np.linalg.norm(embedding_vector, axis=1)),
                        'Std length of embedding vecs: %.6f' % np.std(np.linalg.norm(embedding_vector, axis=1)),
                        ])
        print('Element mean of embedding vec:')
        pprint(np.mean(embedding_vector, axis=0))
    assert len(embedding_keys) == len(embedding_vector), \
        'First dimensions of keys and vectors are not matched.'
    return embedding_keys, embedding_vector


def process(args):
    embedding_keys, embedding_vector = load_embedding_vec(cp.target)

    fu.block_print(['%s\'s # of embedding: %d' % (cp.target, len(embedding_keys)),
                    '%s\'s shape of embedding vec: %s' % (cp.target, str(embedding_vector.shape))])

    print('Length set of words:\n', set([len(k) for k in embedding_keys]))

    embedding_keys, embedding_vector = filter_stat(embedding_keys, embedding_vector, cp.max_length)

    counter_1gram = Counter()
    counter_3gram = Counter()
    counter_6gram = Counter()
    size_set = set()
    gram_embedding_keys = [[] for _ in range(len(embedding_keys))]

    max_size = 0

    # Update counter and divide each word to n-gram.
    for idx, k in enumerate(tqdm(embedding_keys, desc='Counting')):
        counter_1gram.update(k)
        gram_embedding_keys[idx].extend(k)
        three_gram = [k[i:i + 3] for i in range(len(k) - 2)]
        counter_3gram.update(three_gram)
        gram_embedding_keys[idx].extend(three_gram)
        six_gram = [k[i:i + 6] for i in range(len(k) - 5)]
        counter_6gram.update(six_gram)
        gram_embedding_keys[idx].extend(six_gram)
        size_set.add(len(gram_embedding_keys[idx]))
        if max_size < len(gram_embedding_keys[idx]):
            max_size = len(gram_embedding_keys[idx])

    print('Max size of tokens:', max_size)
    print('Size set of tokens:', size_set)
    print('Number of grams:\n',
          '1-gram:', len(counter_1gram), '2-gram:', len(counter_3gram), '3-gram:', len(counter_6gram))

    if not args.debug:
        du.json_dump({'1': counter_1gram, '3': counter_3gram, '6': counter_6gram}, cp.gram_counter_file)
        vocab = list(counter_1gram.keys()) + list(counter_3gram) + list(counter_6gram) + [UNK]
        du.json_dump(vocab, cp.gram_vocab_file)
        gtoi = {gram: idx for idx, gram in enumerate(vocab)}
        encoded_embedding_keys = np.ones((len(embedding_keys), max_size), dtype=np.int64) * (len(gtoi) - 1)
        for idx, k in enumerate(tqdm(gram_embedding_keys, desc='Encoding')):
            encoded_embedding_keys[idx, :len(k)] = [gtoi[gram] for gram in k]
        print(encoded_embedding_keys[:5])
        assert len(encoded_embedding_keys) == len(embedding_vector), \
            'First dimensions of encoded keys and vectors are not matched.'
        start_time = time.time()
        fu.safe_remove(cp.encode_embedding_key_file)
        fu.safe_remove(cp.encode_embedding_vec_file)
        np.save(cp.encode_embedding_key_file, encoded_embedding_keys)
        np.save(cp.encode_embedding_vec_file, embedding_vector)
        print('Saveing processed data with %.3f s' % (time.time() - start_time))


def main():
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='debug')
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()

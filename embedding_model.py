import argparse
import io
import math
import os
import pprint
import time
from collections import Counter
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join, exists
from random import shuffle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.data as tfdata
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
from tqdm import tqdm, trange

import data_utils as du
from config import MovieQAConfig
from model import extract_axis_1

UNK = 'UNK'
RECORD_FILE_PATTERN = join('./embedding_dataset', 'embedding_%05d-of-%05d.tfrecord')
pp = pprint.PrettyPrinter(indent=4, compact=True)
embedding_size = 300
config = MovieQAConfig()


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


def load_glove(filename):
    embedding = {}

    # Read in the data.
    with io.open(filename, 'r', encoding='utf-8') as savefile:
        for i, line in enumerate(tqdm(savefile)):
            tokens = line.rstrip().split(sep=' ', maxsplit=embedding_size)

            word, *entries = tokens

            embedding[word] = [float(x) for x in entries]
            assert len(embedding[word]) == embedding_size, 'Wrong embedding dim.'

    return embedding


def get_initializer():
    if args.initializer == 'identity':
        initializer = tf.identity_initializer()
    elif args.initializer == 'truncated':
        initializer = tf.truncated_normal_initializer(0, args.init_scale)
    elif args.initializer == 'uniform':
        initializer = tf.random_uniform_initializer(-0.08, 0.08)
    elif args.initializer == 'normal':
        initializer = tf.random_normal_initializer(0, args.init_scale)
    elif args.initializer == 'orthogonal':
        initializer = tf.orthogonal_initializer()
    elif args.initializer == 'glorot' or args.initializer == 'xavier':
        initializer = tf.glorot_uniform_initializer()
    else:
        initializer = None
    return initializer


def feature_parser(record):
    features = {
        "vec": tf.FixedLenFeature([embedding_size], tf.float32),
        "word": tf.FixedLenFeature([args.max_length], tf.int64),
        "len": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, features)

    return parsed['vec'], parsed['word'], parsed['len']


class EmbeddingData(object):
    RECORD_FILE_PATTERN_ = join('embedding_dataset', 'embedding_*.tfrecord')

    def __init__(self, batch_size=128, num_thread=16):
        self.batch_size = batch_size
        self.num_example = len(np.load(config.encode_embedding_len_file))
        if args.raw_input:
            pass
        else:
            num_per_shard = int(math.ceil(self.num_example / float(args.num_shards)))
            self.num_example = min(self.num_example, num_per_shard * args.give_shards)
            self.file_names = glob(self.RECORD_FILE_PATTERN_)
            self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
            self.dataset = tf.data.TFRecordDataset(self.file_names_placeholder) \
                .map(feature_parser, num_parallel_calls=num_thread).prefetch(num_thread * batch_size * 4) \
                .shuffle(buffer_size=num_thread * batch_size * 8).apply(tfdata.batch_and_drop_remainder(batch_size))
            self.iterator = self.dataset.make_initializable_iterator()
            self.vec, self.word, self.len = self.iterator.get_next()
            self.vocab = du.load_json(config.char_vocab_file)
            self.vocab_size = len(self.vocab)

    def test(self):
        with tf.Session() as sess:
            sess.run(self.iterator.initializer, feed_dict={
                self.file_names_placeholder: self.file_names
            })

            print(sess.run([self.vec, self.word, self.len]))


class MatricesModel(object):
    def __init__(self, data):
        self.data = data
        initializer = get_initializer()

        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size, embedding_size, embedding_size],
                                           tf.float32, initializer, trainable=True)
        # bias_matrix = tf.get_variable("bias_matrix", [self.data.vocab_size, embedding_size],
        #                               tf.float32, initializer, trainable=True)
        self.char_embedding = tf.transpose(tf.nn.embedding_lookup(embedding_matrix, self.data.word), [1, 0, 2, 3])

        mat_init = tf.get_variable('mat_init', [1, 1, embedding_size], tf.float32, initializer)

        self.mat_init = tf.tile(mat_init, [self.data.batch_size, 1, 1])
        print(self.mat_init.shape)

        self.chain_mul = tf.transpose(tf.scan(lambda a, x: tf.matmul(a, x), self.char_embedding,
                                              initializer=self.mat_init),
                                      [1, 0, 2, 3])
        print(self.chain_mul.shape)

        self.output = extract_axis_1(tf.squeeze(self.chain_mul, 2), self.data.len - 1)
        print(self.output.shape)


class MyConvModel(object):
    def __init__(self, data, char_dim=64, conv_channel=512):
        self.data = data
        initializer = get_initializer()

        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size, char_dim],
                                           tf.float32, initializer, trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)

        conv_output = []
        for i in range(6):
            conv_output.append(layers.conv2d(self.char_embedding, conv_channel,
                                             [i + 1], padding='VALID', activation_fn=None))
            print(conv_output[i].shape)

        for i in range(6):
            conv_output[i] = layers.maxout(conv_output[i], 1, axis=1)
            print(conv_output[i].shape)
        self.conv_concat = tf.squeeze(tf.concat(conv_output, 2), 1)
        self.fc1 = layers.fully_connected(self.conv_concat, embedding_size, activation_fn=None)
        self.output = layers.fully_connected(self.fc1, embedding_size, activation_fn=None)
        print(self.output.shape)


class MyModel(object):
    def __init__(self, data, char_dim=64, hidden_dim=256, num_layers=2):
        start_time = time.time()
        self.data = data
        initializer = get_initializer()

        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size, char_dim],
                                           tf.float32, initializer, trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)
        # self.char_embedding = tf.unstack(self.char_embedding, args.max_length, axis=1)
        # subt_mask = tf.tile(tf.expand_dims(
        #     tf.sequence_mask(self.data.len, args.max_length), axis=2), [1, 1, char_dim])
        # zeros = tf.zeros_like(self.char_embedding)
        # masked_x = tf.where(subt_mask, self.char_embedding, zeros)
        # self.mean_embedding = tf.divide(tf.reduce_sum(masked_x, axis=1),
        #                                 tf.expand_dims(tf.cast(self.data.len, tf.float32), axis=1))
        # print(self.mean_embedding.shape)
        # output_list = []
        # for i in trange(embedding_size):
        #     with tf.variable_scope('embedding_output%d' % i, ):
        total_units = hidden_dim * 16
        if args.rnn_cell == 'GRU':
            cell_fn = partial(tf.nn.rnn_cell.GRUCell, num_units=total_units,
                              kernel_initializer=initializer,
                              bias_initializer=tf.constant_initializer(args.bias_init),
                              activation=tf.nn.leaky_relu)
        elif args.rnn_cell == 'LSTM':
            cell_fn = partial(tf.nn.rnn_cell.LSTMCell, num_units=total_units,
                              initializer=initializer,
                              forget_bias=args.bias_init,
                              activation=tf.nn.leaky_relu)
        elif args.rnn_cell == 'BasicRNN':
            cell_fn = partial(tf.nn.rnn_cell.BasicRNNCell, num_units=total_units,
                              activation=tf.nn.leaky_relu)
        else:
            cell_fn = None

        if args.rnn == 'multi':
            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(num_layers)])
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(num_layers)])
        else:
            lstm_cell_fw = cell_fn()
            lstm_cell_bw = cell_fn()

        self.rnn_outputs, self.rnn_final_state = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, self.char_embedding, self.data.len, dtype=tf.float32)

        # self.rnn_outputs = tf.transpose(tf.stack(self.rnn_outputs), [1, 0, 2])
        # print(self.rnn_outputs.shape)
        # print(self.fw.shape, self.bw.shape)
        # self.rnn_outputs = list(zip(*self.rnn_outputs))
        # self.val_f, self.val_b = extract_axis_1(self.rnn_outputs[:, :, :hidden_dim],
        #                                         self.data.len - 1), \
        #                          extract_axis_1(self.rnn_outputs[:, :, hidden_dim:],
        #                                         tf.zeros_like(self.data.len))
        # print(self.val_f.shape, self.val_b.shape)
        self.fw, self.bw = self.rnn_final_state
        self.fc1 = tf.concat([self.fw, self.bw], axis=1)
        self.fc2 = layers.fully_connected(self.fc1, total_units * 2 // 4, activation_fn=tf.nn.leaky_relu)
        self.fc3 = layers.fully_connected(self.fc2, total_units * 2 // 16, activation_fn=tf.nn.leaky_relu)
        self.output = layers.fully_connected(self.fc3, 300, activation_fn=None)

        print('Elapsed time: %.3f' % (time.time() - start_time))


# identity / truncated / random / orthogonal/ glorot
class EmbeddingModel(object):
    def __init__(self, data, char_embed_dim=100, hidden_dim=256):
        self.data = data
        initializer = get_initializer()

        embedding_matrix = tf.get_variable(
            name="embedding_matrix", initializer=initializer,
            shape=[self.data.vocab_size, char_embed_dim], trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)

        if args.rnn_cell == 'GRU':
            cell_fn = partial(rnn.GRUCell, num_units=hidden_dim,
                              kernel_initializer=initializer,
                              bias_initializer=tf.constant_initializer(args.bias_init))
        elif args.rnn_cell == 'LSTM':
            cell_fn = partial(rnn.CoupledInputForgetGateLSTMCell, num_units=hidden_dim, initializer=initializer)
        elif args.rnn_cell == 'BasicRNN':
            cell_fn = partial(tf.nn.rnn_cell.BasicRNNCell, num_units=hidden_dim)
        else:
            cell_fn = None

        if args.rnn == 'multi':
            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(4)])
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(4)])
        else:
            lstm_cell_fw = cell_fn()
            lstm_cell_bw = cell_fn()

        init_fw_c_state = tf.get_variable("init_fw_c_state", [1, hidden_dim], tf.float32, initializer)
        init_fw_h_state = tf.get_variable("init_fw_h_state", [1, hidden_dim], tf.float32, initializer)
        fw_state = rnn.LSTMStateTuple(tf.tile(init_fw_c_state, [self.data.batch_size, 1]),
                                      tf.tile(init_fw_h_state, [self.data.batch_size, 1]))

        init_bw_c_state = tf.get_variable("init_bw_c_state", [1, hidden_dim], tf.float32, initializer)
        init_bw_h_state = tf.get_variable("init_bw_h_state", [1, hidden_dim], tf.float32, initializer)
        bw_state = rnn.LSTMStateTuple(tf.tile(init_bw_c_state, [self.data.batch_size, 1]),
                                      tf.tile(init_bw_h_state, [self.data.batch_size, 1]))

        self.rnn_outputs, self.rnn_final_state = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, self.char_embedding, self.data.len, fw_state, bw_state, tf.float32)
        # init_fw_state, init_bw_state, tf.float32)
        # self.val_f, self.val_b = extract_axis_1(self.rnn_outputs[0],
        #                                         self.data.len - 1), \
        #                          extract_axis_1(self.rnn_outputs[1],
        #                                         np.zeros(2))
        # of, ob = o
        # of, ob0, ob1 = extract_axis_1(o[0], self.data.len - 1), extract_axis_1(o[1], np.zeros(64)), extract_axis_1(
        #     o[1], self.data.len - 1)
        # sf, sb = s

        self.fw, self.bw = self.rnn_final_state
        if args.rnn == 'multi':
            self.fw = tf.concat([t for t in self.fw], axis=1)
            self.bw = tf.concat([t for t in self.bw], axis=1)
        # self.fw_h, self.bw_h = tf.reshape(self.fw.h, [-1, 16, hidden_dim]),
        # tf.reshape(self.bw.h, [-1, 16, hidden_dim])
        self.fw_h, self.bw_h = self.fw.h, self.bw.h
        self.fc = tf.concat([self.fw_h, self.bw_h], axis=1)
        # self.fct = tf.transpose(self.fc, [1, 0, 2])
        # self.attention = tf.expand_dims(tf.transpose(
        #     layers.fully_connected(layers.flatten(self.fc), 16, tf.nn.softmax,
        #                            biases_initializer=tf.constant_initializer(args.bias_init)), [1, 0]), 2)
        self.attention = tf.expand_dims(tf.transpose(
            layers.fully_connected(self.fc, 16, tf.nn.softmax,
                                   biases_initializer=tf.constant_initializer(args.bias_init)), [1, 0]), 2)

        output_list = []
        for i in range(16):
            with tf.variable_scope('Affine_Transform_%d' % i):
                if args.rnn == 'multi':
                    self.fc = layers.fully_connected(self.fc, 1024)
                fc2 = layers.fully_connected(self.fc, embedding_size, activation_fn=tf.nn.tanh,
                                             biases_initializer=tf.constant_initializer(args.bias_init))
                output_list.append(layers.fully_connected(fc2, embedding_size, activation_fn=None) * self.attention[i])
        print(output_list[0].shape)
        self.output = tf.reduce_sum(tf.stack(output_list, 0), 0)
        print(self.output.shape)
        # self.qq = [of, ob0, ob1, sf, sb]
        # self.qq = [of, ob, sf, sb]

    def test(self):
        with tf.Session() as sess:
            sess.run(self.data.iterator.initializer, feed_dict={
                self.data.file_names_placeholder: self.data.file_names
            })
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            out = sess.run(self.output)
            print(out.shape, out, sep='\n\n\n\n')
            # print(*sess.run(self.qq), sep='\n\nFUCK\n\n')
            # of, ob0, ob1, sf, sb = sess.run(self.qq)
            # print("of, ob0, ob1, sf, sb:", of.shape, ob0.shape, ob1.shape, sf.h.shape, sb.h.shape)
            # print('of == sf:', np.array_equal(of, sf.h))
            # print('ob0 == ob1:', np.array_equal(ob0, ob1))
            # print('ob0 == sb:', np.array_equal(ob0, sb.h))
            # print('ob1 == sb:', np.array_equal(ob1, sb.h))
            # print(of, sf.h, sep='\n\nFUCK\n\n')
            # print(ob0, sb.h, sep='\n\nFUCK\n\n')


def create_one_example(v, w, l):
    feature = {
        "vec": du.to_feature(v),
        "word": du.to_feature(w),
        "len": du.to_feature(l),
    }

    features = tf.train.Features(feature=feature)

    return tf.train.Example(features=features)


def create_one_record(ex_tuple):
    shard_id, example_list = ex_tuple
    output_filename = RECORD_FILE_PATTERN % (shard_id + 1, args.num_shards)
    du.exist_then_remove(output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(example_list)):
            embedding_vec, embedding_word, embedding_word_length = example_list[i]
            example = create_one_example(embedding_vec, embedding_word, embedding_word_length)
            tfrecord_writer.write(example.SerializeToString())


def create_records():
    start_time = time.time()
    embedding_vec = np.load(config.encode_embedding_vec_file)
    embedding_word = np.load(config.encode_embedding_key_file)
    embedding_word_length = np.load(config.encode_embedding_len_file)
    print('Loading file done. Spend %f sec' % (time.time() - start_time))
    du.pprint(['embedding_vec\'s shape:' + str(embedding_vec.shape),
               'embedding_word\'s shape:' + str(embedding_word.shape),
               'embedding_word_length\'s shape:' + str(embedding_word_length.shape)])
    num_per_shard = int(math.ceil(len(embedding_word_length) / float(args.num_shards)))
    example_list = []
    for j in trange(args.num_shards):
        start_ndx = j * num_per_shard
        end_ndx = min((j + 1) * num_per_shard, len(embedding_word_length))
        example_list.append((j, [(embedding_vec[i], embedding_word[i], embedding_word_length[i])
                                 for i in range(start_ndx, end_ndx)]))
    with Pool(8) as pool, tqdm(total=args.num_shards, desc='Tfrecord') as pbar:
        for _ in pool.imap_unordered(create_one_record, example_list):
            pbar.update()


def load_embedding_vec():
    start_time = time.time()
    if args.target == 'glove':
        key_file = config.glove_embedding_key_file
        vec_file = config.glove_embedding_vec_file
        raw_file = config.glove_file
        load_fn = load_glove
    elif args.target == 'w2v':
        key_file = config.w2v_embedding_key_file
        vec_file = config.w2v_embedding_vec_file
        raw_file = config.word2vec_file
        load_fn = load_w2v
    elif args.target == 'fasttext':
        key_file = config.ft_embedding_key_file
        vec_file = config.ft_embedding_vec_file
        raw_file = config.fasttext_file
        load_fn = load_glove
    else:
        key_file = None
        vec_file = None
        raw_file = None
        load_fn = None

    if exists(key_file) and exists(vec_file):
        embedding_keys = du.load_json(key_file)
        embedding_vecs = np.load(vec_file)
    else:
        embedding = load_fn(raw_file)
        embedding_keys = []
        embedding_vecs = np.zeros((len(embedding), embedding_size), dtype=np.float32)
        for i, k in enumerate(embedding.keys()):
            embedding_keys.append(k)
            embedding_vecs[i] = embedding[k]
        du.write_json(embedding_keys, key_file)
        np.save(vec_file, embedding_vecs)

    print('Loading embedding done. %.3f s' % (time.time() - start_time))
    return embedding_keys, embedding_vecs


def filter_stat(embedding_keys, embedding_vecs):
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
            if len(kk) <= args.max_length:
                if keys.get(kk, None):
                    if k.strip().islower():
                        keys[k.strip()] = i
                else:
                    keys[k.lower().strip()] = i
    std = math.sqrt(std / count)
    vecs = embedding_vecs[list(keys.values())]
    embedding_keys, embedding_vecs = list(keys.keys()), vecs

    du.pprint(['Filtered number of embedding: %d' % len(embedding_keys),
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
    # tokenize_qa = du.load_json(config.avail_tokenize_qa_file)
    # subtitle = du.load_json(config.subtitle_file)

    embedding_keys, embedding_vecs = load_embedding_vec()

    du.pprint(['%s\'s # of embedding: %d' % (args.target, len(embedding_keys)),
               '%s\'s shape of embedding vec: %s' % (args.target, str(embedding_vecs.shape))])

    embedding_keys, embedding_vecs = filter_stat(embedding_keys, embedding_vecs)

    embed_char_counter = Counter()
    for k in tqdm(embedding_keys):
        embed_char_counter.update(k)

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

    du.write_json(embed_char_counter, config.embed_char_counter_file)
    # du.write_json(qa_char_counter, config.qa_char_counter_file)

    # count_array = np.array(list(embed_char_counter.values()), dtype=np.float32)
    # m, v, md, f = np.mean(count_array), np.std(count_array), np.median(count_array), np.percentile(count_array, 95)
    # print(m, v, md, f)
    #
    # above_mean = dict(filter(lambda item: item[1] > f, embed_char_counter.items()))
    # below_mean = dict(filter(lambda item: item[1] < f, embed_char_counter.items()))
    # below_occur = set(filter(lambda k: k in qa_char_counter, below_mean.keys()))
    # final_set = below_occur.union(set(above_mean.keys()))
    # du.write_json(list(final_set) + [UNK], config.char_vocab_file)
    vocab = list(embed_char_counter.keys()) + [UNK]
    print('Filtered vocab:', vocab)
    du.write_json(vocab, config.char_vocab_file)
    # vocab = du.load_json(config.char_vocab_file)
    encode_embedding_keys = np.ones((len(embedding_keys), args.max_length), dtype=np.int64) * (len(vocab) - 1)
    length = np.zeros(len(embedding_keys), dtype=np.int64)
    for i, k in enumerate(tqdm(embedding_keys, desc='Encoding...')):
        encode_embedding_keys[i, :len(k)] = [
            vocab.index(ch) if ch in vocab else vocab.index(UNK)
            for ch in k
        ]
        assert all([idx < len(vocab) for idx in encode_embedding_keys[i]]), \
            "Wrong index!"
        length[i] = len(k)
    du.pprint(['Shape of encoded key: %s' % str(encode_embedding_keys.shape),
               'Shape of encoded key length: %s' % str(length.shape)])
    start_time = time.time()
    du.exist_then_remove(config.encode_embedding_key_file)
    du.exist_then_remove(config.encode_embedding_len_file)
    du.exist_then_remove(config.encode_embedding_vec_file)
    np.save(config.encode_embedding_key_file, encode_embedding_keys)
    np.save(config.encode_embedding_len_file, length)
    np.save(config.encode_embedding_vec_file, embedding_vecs)
    print('Saveing processed data with %.3f s' % (time.time() - start_time))


def inspect():
    data = EmbeddingData(2)
    model = MatricesModel(data)
    for v in tf.global_variables():
        print(v, v.shape)
        # # norm_y, norm_y_ = tf.nn.l2_normalize(model.output, 1), tf.nn.l2_normalize(data.vec, 1)
        # # loss = tf.losses.cosine_distance(norm_y_, norm_y, 1)
    config_ = tf.ConfigProto(allow_soft_placement=True, )
    config_.gpu_options.allow_growth = True

    with tf.Session(config=config_) as sess:
        sess.run(data.iterator.initializer, feed_dict={
            data.file_names_placeholder: data.file_names
        })
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        outs = sess.run(model.output)
        print(outs)
        print(outs.shape)

        #     outs = sess.run(model.output)
        #     print(outs)
        #     print(outs.shape)
        # print(*[t.shape for t in outs])
        #     fw, bw, output = sess.run([model.fw, model.bw, model.output])
        #     print(fw.shape, bw.shape, output.shape)
        #     print(output)
        # out = sess.run(model.output)
        # print(out.shape)
        # print(out)
        # print(*sess.run([model.highway_like, model.output]), sep='\n\n')

        # l, y, y_ = sess.run([loss, norm_y, norm_y_])
        # print('Loss: %.4f' % l)
        # print('Normalized output\'s shape:')
        # pp.pprint(y.shape)
        # print('Normalized label\'s shape:')
        # pp.pprint(y_.shape)
        #     fw, bw, f, b = sess.run([model.fw, model.bw, model.val_f, model.val_b])
        #     print(f.shape, b.shape)
        #     print(fw.shape, bw.shape)
        #     print(np.array_equal(fw, f))
        #     print(np.array_equal(bw, b))
        #     print(fw, '=' * 87, bw, sep='\n')
        # embedding_keys, embedding_vecs = load_embedding_vec()
        #
        # du.pprint(['w2v\'s # of embedding: %d' % len(embedding_keys),
        #            'w2v\'s shape of embedding vec: ' + str(embedding_vecs.shape)])
        #
        # filter_stat(embedding_keys, embedding_vecs)

        # vocab = du.load_json(config.char_vocab_file)
        # length = np.load(config.encode_embedding_len_file)
        # vecs = np.load(config.encode_embedding_vec_file)
        # lack = [ch for ch in string.ascii_lowercase + string.digits if ch not in vocab]
        #
        # print(lack)
        # print(vocab)
        # print(max(length))
        # print(vecs.shape)


def get_loss(name, data, model):
    if name == 'mse':
        loss = tf.losses.mean_squared_error(data.vec, model.output)
    elif name == 'abs':
        loss = tf.losses.absolute_difference(data.vec, model.output)
    elif name == 'l2':
        loss = tf.losses.compute_weighted_loss(tf.norm(data.vec - model.output, axis=1))
    elif name == 'cos':
        loss = tf.losses.cosine_distance(tf.nn.l2_normalize(data.vec, 1),
                                         tf.nn.l2_normalize(model.output, 1), 1)
    elif name == 'huber':
        loss = tf.losses.huber_loss(data.vec, model.output)
    elif name == 'mpse':
        loss = tf.losses.mean_pairwise_squared_error(data.vec, model.output)
    else:
        loss = 0
    return loss


def main():
    data = EmbeddingData(args.batch_size)
    if args.model == 'myconv':
        model = MyConvModel(data, char_dim=args.char_embed_dim, conv_channel=args.conv_channel)
    elif args.model == 'myrnn':
        model = MyModel(data, char_dim=args.char_embed_dim, hidden_dim=args.hidden_dim, num_layers=args.nlayers)
    elif args.model == 'mimick':
        model = EmbeddingModel(data, char_embed_dim=args.char_embed_dim, hidden_dim=args.hidden_dim)
    elif args.model == 'matrice':
        model = MatricesModel(data)
    else:
        model = None

    args_dict = vars(args)
    pairs = [(k, str(args_dict[k])) for k in sorted(args_dict.keys())
             if args_dict[k] != parser.get_default(k) and not isinstance(args_dict[k], bool)]
    if pairs:
        exp_name = '_'.join(['.'.join(pair) for pair in pairs])
    else:
        exp_name = '.'.join(('learning_rate', str(args.learning_rate)))

    log_dir = join(config.log_dir, 'embedding_log', exp_name)
    checkpoint_dir = join(config.checkpoint_dir, 'embedding_checkpoint', exp_name)
    checkpoint_name = join(checkpoint_dir, 'embedding')

    if args.reset:
        if exists(checkpoint_dir):
            os.system('rm -rf %s' % os.path.join(checkpoint_dir, '*'))
        if os.path.exists(log_dir):
            os.system('rm -rf %s' % os.path.join(log_dir, '*'))
    du.exist_make_dirs(log_dir)
    du.exist_make_dirs(checkpoint_dir)

    loss = get_loss(args.loss, data, model) + get_loss(args.sec_loss, data, model)

    baseline = tf.losses.mean_squared_error(data.vec, model.output)

    global_step = tf.train.get_or_create_global_step()
    total_step = int(math.floor(data.num_example / args.batch_size))
    learning_rate = tf.train.exponential_decay(args.learning_rate,
                                               global_step,
                                               args.decay_epoch * total_step,
                                               args.decay_rate,
                                               staircase=True)
    if args.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif args.optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        optimizer = None

    grads_and_vars = optimizer.compute_gradients(loss)
    gradients, variables = list(zip(*grads_and_vars))
    for var in variables:
        print(var.name, var.shape)

    if not args.model == 'myconv':
        capped_grads_and_vars = [(tf.clip_by_norm(gv[0], args.clip_norm), gv[1]) for gv in grads_and_vars]
    else:
        capped_grads_and_vars = grads_and_vars

    train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step)
    saver = tf.train.Saver(tf.global_variables(), )

    # Summary
    train_gv_summaries = []
    for idx, var in enumerate(variables):
        train_gv_summaries.append(tf.summary.histogram('gradient/' + var.name, gradients[idx]))
        train_gv_summaries.append(tf.summary.histogram(var.name, var))

    train_gv_summaries_op = tf.summary.merge(train_gv_summaries)

    train_summaries = [
        tf.summary.scalar('loss', loss),
        tf.summary.scalar('baseline', baseline),
        tf.summary.scalar('learning_rate', learning_rate)
    ]
    train_summaries_op = tf.summary.merge(train_summaries)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    if args.checkpoint_file:
        checkpoint_file = args.checkpoint_file
    else:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    # restore_fn = (lambda _sess: saver.restore(_sess, checkpoint_file)) \
    #     if checkpoint_file else (lambda _sess: _sess.run(init_op))

    # sv = tf.train.Supervisor(logdir=log_dir, summary_op=None,
    #                          init_fn=restore_fn, save_model_secs=0,
    #                          saver=saver, global_step=global_step)

    config_ = tf.ConfigProto(allow_soft_placement=True, )
    config_.gpu_options.allow_growth = True
    # with sv.managed_session() as sess:
    with tf.Session(config=config_) as sess, tf.summary.FileWriter(log_dir) as sw:
        sess.run(init_op)

        if checkpoint_file:
            saver.restore(sess, checkpoint_file)

        def save():
            saver.save(sess, checkpoint_name, tf.train.global_step(sess, global_step))

        def save_sum(summary_):
            sw.add_summary(summary_, tf.train.global_step(sess, global_step))

        # Training loop
        def train_loop(epoch_, y=0, y_=0):
            shuffle(data.file_names)
            sess.run(data.iterator.initializer, feed_dict={
                data.file_names_placeholder: data.file_names[:args.give_shards],
            })
            step = tf.train.global_step(sess, global_step)
            print("Training Loop Epoch %d" % epoch_)
            step = step % total_step
            pbar = trange(step, total_step)
            for _ in pbar:
                try:
                    _, l, bl, step, y, y_, gv_summary, summary = sess.run(
                        [train_op, loss, baseline, global_step,
                         model.output, data.vec,
                         train_gv_summaries_op, train_summaries_op])
                    if step % max((total_step // 100), 10) == 0:
                        save_sum(summary)
                        save_sum(gv_summary)
                    if step % max((total_step // 10), 100) == 0:
                        save()
                    pbar.set_description(
                        '[%s/%s] step: %d loss: %.3f base: %.3f' % (epoch_, args.epoch, step, l, bl))
                except KeyboardInterrupt:
                    save()
                    print()
                    return True

            print("Training Loop Epoch %d Done..." % epoch_)
            print(y_, y, sep='\n\n\n')
            time.sleep(10)
            save()
            return False

        now_epoch = tf.train.global_step(sess, global_step) // total_step + 1
        for epoch in range(now_epoch, args.epoch + 1):
            if train_loop(epoch):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Program Mode
    parser.add_argument('--process', action='store_true',
                        help='Process the data which creating tfrecords needs.')
    parser.add_argument('--inspect', action='store_true', help='Inspect the data stat.')
    parser.add_argument('--tfrecord', action='store_true', help='Create tfrecords.')
    # Embedding Target
    parser.add_argument('--target', default='glove', help='Learning target of word embedding.')
    parser.add_argument('--num_shards', default=128, help='Number of tfrecords.', type=int)
    parser.add_argument('--max_length', default=12, help='Maximal word length.', type=int)
    parser.add_argument('--raw_input', action='store_ture', help='Use raw data as input without tfreord.')
    # Training Setting
    parser.add_argument('--reset', action='store_true', help='Reset the experiment.')
    parser.add_argument('--give_shards', default=1, help='Number of training shards given', type=int)
    parser.add_argument('--checkpoint_file', default=None, help='Checkpoint file')
    parser.add_argument('--learning_rate', default=1E-4, help='Initial learning rate.', type=float)
    parser.add_argument('--batch_size', default=2, help='Batch size of training.', type=int)
    parser.add_argument('--dropout_prob', default=1.0, help='Probability of dropout.', type=float)
    parser.add_argument('--char_embed_dim', default=64, help='Dimension of char embedding', type=int)
    parser.add_argument('--hidden_dim', default=256, help='Dimension of hidden state.', type=int)
    parser.add_argument('--conv_channel', default=512, help='Output channel of convolution layer.', type=int)
    parser.add_argument('--epoch', default=200, help='Training epochs', type=int)
    parser.add_argument('--decay_epoch', default=2, help='Span of epochs at decay.', type=int)
    parser.add_argument('--decay_rate', default=0.87, help='Decay rate.', type=float)
    parser.add_argument('--optimizer', default='adam', help='Training policy (adam / momentum / sgd / rms).')
    parser.add_argument('--loss', default='mse', help='Fist loss function. (mse / cos / abs / l2 / huber / mpse)')
    parser.add_argument('--sec_loss', default=None, help='Second loss function. (mse / cos / abs / l2 / huber / mpse)')
    parser.add_argument('--clip_norm', default=1.0, help='Norm value of gradient clipping.', type=float)
    parser.add_argument('--initializer', default='truncated',
                        help='Initializer of weight.\n(identity / truncated / random / orthogonal / glorot)')
    parser.add_argument('--rnn', default='single', help='Multi / Single-layer rnn.')
    parser.add_argument('--nlayers', default=2, help='Number of Layers in rnn.', type=int)
    parser.add_argument('--rnn_cell', default='LSTM', help='RNN cell type. (GRU / LSTM / BasicRNN)')
    parser.add_argument('--bias_init', default=1.0, help='RNN cell bias initialization value.', type=float)
    parser.add_argument('--model', default='matrice', help='Model modality.')
    parser.add_argument('--init_scale', default=0.15, help='Initial value of weight.', type=float)

    args = parser.parse_args()
    if args.process:
        process()
    if args.tfrecord:
        create_records()
    if args.inspect:
        inspect()
    if not (args.process or args.tfrecord or args.inspect):
        main()
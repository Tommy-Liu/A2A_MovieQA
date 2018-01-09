import itertools
import math
import numbers
import os
import pprint
import random
import shutil
import sys
import time
from collections import Counter
from glob import glob
from multiprocessing import Pool
from os.path import join, exists
from random import shuffle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.data as tf_data
from tqdm import tqdm, trange

from config import MovieQAConfig
from embed.args import args_parse
from utils import data_utils as du
from utils import func_utils as fu
from utils.model_utils import get_opt, get_loss

UNK = 'UNK'
RECORD_FILE_PATTERN = join('./embedding_dataset', 'embedding_%s%05d-of-%05d.tfrecord')
pp = pprint.PrettyPrinter(indent=4, compact=True)
embedding_size = 300
config = MovieQAConfig()


def feature_parser(record):
    features = {
        "vec": tf.FixedLenFeature([embedding_size], tf.float32),
        "word": tf.FixedLenFeature([args.max_length], tf.int64),
        "len": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, features)

    return parsed['vec'], parsed['word'], parsed['len']


class EmbeddingData(object):
    RECORD_FILE_PATTERN_ = join('embedding_dataset', 'embedding_%s*.tfrecord')

    def __init__(self, batch_size=128, num_thread=16, num_given=sys.maxsize,
                 use_length=12, raw_input=False):
        '''
        initial target:
        self.iterator.initializer
        feed_dict:
        file_names_placeholder
        '''
        start_time = time.time()
        self.raw_len = np.load(config.encode_embedding_len_file)
        self.batch_size = batch_size
        # TODO(tommy8054): Raw input from numpy array. Is it necessary to implement this? QQ
        if raw_input:
            # Filter instances
            length = np.load(config.encode_embedding_len_file)
            vec = np.load(config.encode_embedding_vec_file)[length <= use_length]
            word = np.load(config.encode_embedding_key_file)[length <= use_length]
            length = length[length <= use_length]
            vec = vec[:min(len(vec), num_given)]
            word = word[:min(len(word), num_given)]
            length = length[:min(len(length), num_given)]
            # Build up input pipeline
            self.load = {'vec': vec, 'word': word, 'len': length}
            self.vec, self.word, self.len = tf.placeholder(tf.float32, [None, embedding_size], 'vec'), \
                                            tf.placeholder(tf.int64, [None, args.max_length], 'word'), \
                                            tf.placeholder(tf.int64, [None], 'length')

            self.vec_temp, self.word_temp, self.len_temp = tf.placeholder(tf.float32, [None, embedding_size],
                                                                          'vec_temp'), \
                                                           tf.placeholder(tf.int64, [None, args.max_length],
                                                                          'word_temp'), \
                                                           tf.placeholder(tf.int64, [None], 'length_temp')

            self.dataset = tf_data.Dataset.from_tensor_slices(
                (self.vec_temp, self.word_temp, self.len_temp)) \
                .prefetch(num_thread * batch_size * 4) \
                .shuffle(buffer_size=num_thread * batch_size * 8) \
                .apply(tf_data.batch_and_drop_remainder(batch_size))
            self.iterator = self.dataset.make_initializable_iterator()
            self.input = self.iterator.get_next()
        else:
            self.num_each_len = [np.sum(self.raw_len == (i + 1), dtype=np.int64) for i in range(args.max_length)]
            self.num_example = min(len(self.raw_len), num_given)
            # Use floor instead of ceil because we drop last batch.
            self.total_step = int(math.floor(self.num_example / self.batch_size))
            self.file_names = glob(self.RECORD_FILE_PATTERN_ % 'length_')
            self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
            self.dataset = tf.data.TFRecordDataset(self.file_names_placeholder) \
                .shuffle(buffer_size=160) \
                .map(feature_parser, num_parallel_calls=num_thread) \
                .prefetch(2000) \
                .shuffle(buffer_size=1000) \
                .apply(tf_data.batch_and_drop_remainder(batch_size)) \
                .repeat()
            self.iterator = self.dataset.make_initializable_iterator()
            self.vec, self.word, self.len = self.iterator.get_next()
        self.vocab = du.json_load(config.char_vocab_file)
        self.vocab_size = len(self.vocab)
        print('Data Loading Finished with %.3f s.' % (time.time() - start_time))

    def test(self):
        with tf.Session() as sess:
            sess.run(self.iterator.initializer, feed_dict={
                self.file_names_placeholder: self.file_names
            })

            print(sess.run([self.vec, self.word, self.len]))

    def get_records(self, length):
        self.num_example = sum([self.num_each_len[i - 1] for i in length])
        self.total_step = int(math.floor(self.num_example / self.batch_size))
        return [n for n in self.file_names for l in length if 'length_%d_' % l in n]



def create_one_example(v, w, l):
    feature = {
        "vec": du.to_feature(v),
        "word": du.to_feature(w),
        "len": du.to_feature(l),
    }

    features = tf.train.Features(feature=feature)

    return tf.train.Example(features=features)


def create_one_record(ex_tuple):
    output_filename, example_list = ex_tuple
    fu.exist_then_remove(output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(example_list)):
            embedding_vec, embedding_word, embedding_word_length = example_list[i]
            example = create_one_example(embedding_vec, embedding_word, embedding_word_length)
            tfrecord_writer.write(example.SerializeToString())


def create_records():
    '''
    Inputs list contains tuples <- (record name, list of tuple <- (vec, word, length))
    It is for multiprocessing.
    :return:
    '''
    start_time = time.time()
    embedding_vec = np.load(config.encode_embedding_vec_file)
    embedding_word = np.load(config.encode_embedding_key_file)
    embedding_word_length = np.load(config.encode_embedding_len_file)
    inputs = []
    print('Loading file done. Spend %.3f sec' % (time.time() - start_time))
    fu.block_print(['embedding_vec\'s shape:' + str(embedding_vec.shape),
                    'embedding_word\'s shape:' + str(embedding_word.shape),
                    'embedding_word_length\'s shape:' + str(embedding_word_length.shape)])
    if args.sorted:
        for i in trange(args.max_length):
            vec = embedding_vec[embedding_word_length == (i + 1)]
            word = embedding_word[embedding_word_length == (i + 1)]
            length = embedding_word_length[embedding_word_length == (i + 1)]
            num_shards = int(math.ceil(len(length) / float(args.num_per_shard)))
            for j in trange(num_shards):
                start_ndx = j * args.num_per_shard
                end_ndx = min((j + 1) * args.num_per_shard, len(length))
                inputs.append((RECORD_FILE_PATTERN % ('length_%d_' % (i + 1), j + 1, num_shards),
                               [(vec[k], word[k], length[k]) for k in range(start_ndx, end_ndx)]))
    else:
        num_per_shard = int(math.ceil(len(embedding_word_length) / float(args.num_shards)))
        for j in trange(args.num_shards):
            start_ndx = j * num_per_shard
            end_ndx = min((j + 1) * num_per_shard, len(embedding_word_length))
            inputs.append((RECORD_FILE_PATTERN % ('', j + 1, args.num_shards),
                           [(embedding_vec[i], embedding_word[i], embedding_word_length[i])
                            for i in range(start_ndx, end_ndx)]))
    with Pool(8) as pool, tqdm(total=len(inputs), desc='Tfrecord') as pbar:
        for _ in pool.imap_unordered(create_one_record, inputs):
            pbar.update()


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
    # tokenize_qa = du.json_load(config.avail_tokenize_qa_file)
    # subtitle = du.json_load(config.subtitle_file)
    embedding_keys, embedding_vecs = du.load_embedding_vec(args.target)

    fu.block_print(['%s\'s # of embedding: %d' % (args.target, len(embedding_keys)),
                    '%s\'s shape of embedding vec: %s' % (args.target, str(embedding_vecs.shape))])

    embedding_keys, embedding_vecs = filter_stat(embedding_keys, embedding_vecs, args.max_length)

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
    for l in range(2, args.max_length + 1):
        target = [k for k in frequency.keys() if len(k) == l]
        total = np.sum([frequency[t] for t in target])

        probability.update({t: frequency[t] / total for t in target})

    # traverse(root)
    # print(root)
    if not args.debug:
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

        du.json_dump(embed_char_counter, config.embed_char_counter_file)
        # du.json_dump(qa_char_counter, config.qa_char_counter_file)

        # count_array = np.array(list(embed_char_counter.values()), dtype=np.float32)
        # m, v, md, f = np.mean(count_array), np.std(count_array), np.median(count_array), np.percentile(count_array, 95)
        # print(m, v, md, f)
        #
        # above_mean = dict(filter(lambda item: item[1] > f, embed_char_counter.items()))
        # below_mean = dict(filter(lambda item: item[1] < f, embed_char_counter.items()))
        # below_occur = set(filter(lambda k: k in qa_char_counter, below_mean.keys()))
        # final_set = below_occur.union(set(above_mean.keys()))
        # du.json_dump(list(final_set) + [UNK], config.char_vocab_file)
        vocab = list(embed_char_counter.keys()) + [UNK]
        print('Filtered vocab:', vocab)
        du.json_dump(vocab, config.char_vocab_file)
        # vocab = du.json_load(config.char_vocab_file)
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
        fu.block_print(['Shape of encoded key: %s' % str(encode_embedding_keys.shape),
                        'Shape of encoded key length: %s' % str(length.shape)])
        start_time = time.time()
        fu.exist_then_remove(config.encode_embedding_key_file)
        fu.exist_then_remove(config.encode_embedding_len_file)
        fu.exist_then_remove(config.encode_embedding_vec_file)
        np.save(config.encode_embedding_key_file, encode_embedding_keys)
        np.save(config.encode_embedding_len_file, length)
        np.save(config.encode_embedding_vec_file, embedding_vecs)
        print('Saveing processed data with %.3f s' % (time.time() - start_time))


def inspect():
    manager = EmbeddingTrainManager(args, parser)
    manager.train()
    # manager.inject_param(val={'max_length': random.randint(1, 12), 'baseline': random.random()})
    pp.pprint(du.json_load(manager.param_file))
    # manager.train()
    # data = EmbeddingData(2)
    # model = MatricesModel(data)
    # # for v in tf.global_variables():
    # #     print(v, v.shape)
    # #     # # norm_y, norm_y_ = tf.nn.l2_normalize(model.output, 1), tf.nn.l2_normalize(data.vec, 1)
    # #     # # loss = tf.losses.cosine_distance(norm_y_, norm_y, 1)
    # config_ = tf.ConfigProto(allow_soft_placement=True, )
    # config_.gpu_options.allow_growth = True
    # with tf.Session(config=config_) as sess:
    #     sess.run([data.iterator.initializer,
    #               tf.global_variables_initializer(),
    #               tf.local_variables_initializer()],
    #              feed_dict={
    #                  data.file_names_placeholder: data.get_records([1, 2]),
    #              })
    #     cn, y = 0, 0
    #     start_time = time.time()
    #     while True:
    #         try:
    #             y = sess.run(model.output)
    #             cn += 1
    #         except tf.errors.OutOfRangeError:
    #             break
    #     print(y)
    #     print(cn)
    #     print('%.3f s' % (time.time() - start_time))
    #     sess.run(data.iterator.initializer, feed_dict={
    #         data.word_temp: data.load['word'],
    #         data.vec_temp: data.load['vec'],
    #         data.len_temp: data.load['len']
    #     })
    #     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #     inputs = sess.run(data.input)
    #     print(sess.run(model.output, feed_dict={
    #         data.vec: inputs[0],
    #         data.word: inputs[1],
    #         data.len: inputs[2]
    #     }))

    #     outs = sess.run(model.output)
    #     print(outs)
    #     print(outs.shape)

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
    # embedding_keys, embedding_vecs = du.load_embedding_vec()
    #
    # du.pprint(['w2v\'s # of embedding: %d' % len(embedding_keys),
    #            'w2v\'s shape of embedding vec: ' + str(embedding_vecs.shape)])
    #
    # filter_stat(embedding_keys, embedding_vecs)

    # vocab = du.json_load(config.char_vocab_file)
    # length = np.load(config.encode_embedding_len_file)
    # vecs = np.load(config.encode_embedding_vec_file)
    # lack = [ch for ch in string.ascii_lowercase + string.digits if ch not in vocab]
    #
    # print(lack)
    # print(vocab)
    # print(max(length))
    # print(vecs.shape)


# TODO(tommy8054): Well... A little bit lazy to implement this... Maybe later?
class EmbeddingTrainManager(object):
    perturbation = {
        'learning_rate': lambda a, i: a * (10 ** i),
        'init_mean': lambda a, i: a + (0.1 * i),
        'init_scale': lambda a, i: a + (0.025 * i)
    }
    diff = {
        'learning_rate': lambda a, b: round(math.log10(b / a)),
        'init_mean': lambda a, b: round((b - a) * 10),
        'init_scale': lambda a, b: round((b - a) * 40)
    }
    target = ['learning_rate', 'init_mean', 'init_scale']

    def __init__(self, arguments, args_parser):
        # Initialize all constant parameters,
        # including paths, hyper-parameters, model name...etc.
        self.args = arguments
        self.args_dict = self.args.__dict__
        self.val_dict = {'baseline': 0, 'max_length': 1, 'lock': True}
        self.args_parser = args_parser

        if len(sys.argv) == 1 or self.args.mode == 'inspect':
            self.search_param()
        else:
            self.retrieve_param(self.param_file)

        if self.args.reset:
            self.remove()
        fu.exist_make_dirs(self.log_dir)
        fu.exist_make_dirs(self.checkpoint_dir)
        if args.checkpoint_file:
            self.checkpoint_file = args.checkpoint_file
        else:
            self.checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        # Lock the current experiment.
        self.lock_exp()
        try:
            # tensorflow pipeline
            tf.reset_default_graph()
            self.data = EmbeddingData(self.args.batch_size)
            with tf.variable_scope('model'):
                self.model = self.get_model()
            self.loss = get_loss(self.args.loss, self.data, self.model) + \
                        get_loss(self.args.sec_loss, self.data, self.model) + \
                        tf.reduce_mean(tf.abs(tf.norm(self.model.output, axis=1) - 1))

            self.baseline = get_loss('mse', self.data, self.model)

            self.global_step = tf.train.get_or_create_global_step()
            self.local_step = tf.get_variable('local_step', dtype=tf.int32, initializer=0, trainable=False)

            self.lr_placeholder = tf.placeholder(tf.float32, name='lr_placeholder')
            self.step_placeholder = tf.placeholder(tf.int64, name='step_placeholder')
            self.dr_placeholder = tf.placeholder(tf.float32, name='dr_placeholder')
            self.learning_rate = tf.train.exponential_decay(self.lr_placeholder,
                                                            self.local_step,
                                                            self.args.decay_epoch * self.step_placeholder,
                                                            self.dr_placeholder,
                                                            staircase=True)
            with tf.variable_scope('optimizer'):
                self.optimizer = get_opt(self.args.optimizer, self.learning_rate)

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            gradients, variables = list(zip(*grads_and_vars))
            for var in variables:
                print(var.name, var.shape)

            if not args.model == 'myconv':
                capped_grads_and_vars = [(tf.clip_by_norm(gv[0], args.clip_norm), gv[1]) for gv in grads_and_vars]
            else:
                capped_grads_and_vars = grads_and_vars

            self.train_op = tf.group(self.optimizer.apply_gradients(capped_grads_and_vars, self.global_step),
                                     tf.assign(self.local_step, self.local_step + 1))
            self.saver = tf.train.Saver(tf.global_variables(), )

            # Summary
            for idx, var in enumerate(variables):
                tf.summary.histogram('gradient/' + var.name, gradients[idx])
                tf.summary.histogram(var.name, var)

            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('baseline', self.baseline),
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.train_summaries_op = tf.summary.merge_all()

            self.init_op = tf.variables_initializer(tf.global_variables() + tf.local_variables())
            self.init_feed_dict = {self.model.init_mean: self.args.init_mean,
                                   self.model.init_stddev: self.args.init_scale}

            pp.pprint(tf.global_variables())
        finally:
            self.unlock_exp()

    def lock_exp(self):
        self.inject_param(val={'lock': True})

    def unlock_exp(self):
        self.inject_param(val={'lock': False})

    @property
    def exp_name(self):
        args_dict = vars(self.args)
        # Filter the different value.
        pairs = []
        for k in sorted(args_dict.keys()):
            default = self.args_parser.get_default(k)
            if default is not None \
                    and args_dict[k] != default \
                    and not isinstance(args_dict[k], bool):
                if isinstance(args_dict[k], numbers.Number):
                    pairs.append((k, '%.2E' % args_dict[k]))
                else:
                    pairs.append((k, str(args_dict[k])))
        # Compose the experiment name.
        if pairs:
            return '$'.join(['.'.join(pair) for pair in pairs])
        else:
            return '.'.join(['learning_rate', '%.2E' % self.args.learning_rate])

    @property
    def log_dir(self):
        return join(config.log_dir, 'embedding_log', self.exp_name)

    @property
    def checkpoint_dir(self):
        return join(config.checkpoint_dir, 'embedding_checkpoint', self.exp_name)

    @property
    def checkpoint_name(self):
        return join(self.checkpoint_dir, 'embedding')

    @property
    def param_file(self):
        return join(self.log_dir, '%s.json' % self.exp_name)

    @property
    def train_feed_dict(self):
        return {
            self.lr_placeholder: self.args.learning_rate,
            self.dr_placeholder: self.args.decay_rate,
            self.step_placeholder: self.args.decay_epoch * self.data.total_step
        }

    def search_param(self):
        exp_paths = glob(join(config.log_dir, 'embedding_log', '**', '*.json'), recursive=True)
        exps = []
        for p in exp_paths:
            d = du.json_load(p)
            exp = {k: d['args'][k] for k in self.target}
            exp.update(d['val'])
            exps.append(exp)

        for idx, exp in enumerate(exps):
            print('%d.' % (idx + 1))
            pp.pprint(exp)

        choice = int(input('Choose one (0 or nothing=random perturbation):') or 0)
        while choice and exps[choice - 1]['lock']:
            choice = int(input('Your choice is locked. Please try the other one:') or 0)

        if choice:
            pp.pprint(exps[choice - 1])
            for k in self.target:
                self.args_dict[k] = exps[choice - 1][k]
            for k in self.val_dict.keys():
                self.val_dict[k] = exps[choice - 1][k]
        else:
            grid = set(tuple(self.diff[k](self.args_dict[k], exp[k])
                             for k in self.target)
                       for exp in exps)
            base = set(list(itertools.product(range(-1, 2), repeat=len(self.target))))
            chosen = base.difference(grid)
            print(len(grid), len(base), len(chosen))
            for idx, p in enumerate(list(iter(chosen))[0]):
                param = self.target[idx]
                self.args_dict[param] = self.perturbation[param](self.args_dict[param], p)
                print(param + ':', self.args_dict[param])

        self.args.reset = bool(input('Reset? (any input=True):'))

    def inject_param(self, arg=None, val=None):
        if arg:
            for k in arg:
                if k in self.args_dict:
                    self.args_dict[k] = arg[k]
        if val:
            for k in val:
                if k in self.val_dict:
                    self.val_dict[k] = val[k]
        d = {'args': self.args_dict, 'val': self.val_dict}
        print('1*')
        pp.pprint(d)
        print('2*')
        du.json_dump(d, self.param_file)
        print('3*')

    def retrieve_param(self, file_name=None):
        if exists(file_name):
            d = du.json_load(file_name)
            self.args_dict, self.val_dict = d['args'], d['val']

    def remove(self):
        if exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        if exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def train(self):
        config_ = tf.ConfigProto(allow_soft_placement=True, )
        config_.gpu_options.allow_growth = True
        with tf.Session(config=config_) as sess, tf.summary.FileWriter(self.log_dir) as sw:
            # Initialize all variables
            sess.run(self.init_op, feed_dict=self.init_feed_dict)
            if self.checkpoint_file:
                print(self.checkpoint_file)
                self.saver.restore(sess, self.checkpoint_file)

            max_length = self.val_dict['max_length']
            # Minimize loss until each of length commits.
            self.lock_exp()
            try:
                while max_length < 13:
                    delta, prev_loss, step = 1, 0, 1
                    try:
                        if not args.debug:
                            # Give instances of current length.
                            sess.run(self.data.iterator.initializer, feed_dict={
                                self.data.file_names_placeholder: self.data.get_records(
                                    list(range(1, max_length + 1)))})
                            # Minimize loss of current length until loss unimproved.
                            while abs(delta) > 1E-8:
                                _, bl, step, summary, loss = sess.run(
                                    [self.train_op, self.baseline, self.global_step, self.train_summaries_op,
                                     self.loss],
                                    feed_dict=self.train_feed_dict)
                                if step % 10 == 0:
                                    sw.add_summary(summary, tf.train.global_step(sess, self.global_step))
                                    print('|-- {:<15}: {:>30}'.format('total_step', self.data.total_step))
                                    print('\n'.join(['|-- {:<15}: {:>30.2E}'
                                                    .format(k, self.args_dict[k]) for k in self.target]))
                                if step % 100 == 0:
                                    self.saver.save(sess, self.checkpoint_name,
                                                    tf.train.global_step(sess, self.global_step))
                                print('[{:>2}/{:>2}] step: {:>6} delta: {:>+10.2E} base: {:>+10.2E} loss: {:>+10.2E}'
                                      .format(max_length, self.args.max_length, step, delta, bl, loss))
                                delta = bl - prev_loss
                                prev_loss = bl
                            # Increment the current length if loss is lower than threshold,
                            # or reset to 1 to search for other possibility.
                            pp.pprint(sess.run([self.data.vec, self.model.output]))
                            print('Length %d loss minimization done.' % max_length)
                        else:
                            bl = random.random() * 1E-3

                        if bl > 1E-1:
                            sess.run(self.init_op, feed_dict=self.init_feed_dict)
                            max_length = 1
                            print('Length %d fail. Reset all.' % max_length)
                        else:
                            print('1-')
                            sess.run(self.local_step.initializer)
                            print('2-')
                            #  Numpy type object is not JSON serializable. Since that, apply float to bl.
                            self.inject_param(val={'max_length': max_length, 'baseline': float(bl)})
                            print('3-')
                            max_length += 1
                            print('4-')
                            print('Length %d commits.' % max_length)
                            print('5-')
                    except KeyboardInterrupt:
                        self.saver.save(sess, self.checkpoint_name, tf.train.global_step(sess, self.global_step))
                        break
            finally:
                self.unlock_exp()

    def get_model(self):
        if self.args.model == 'myconv':
            model = MyConvModel(self.data, char_dim=self.args.char_dim, conv_channel=self.args.conv_channel)
        elif self.args.model == 'myrnn':
            model = MyModel(self.data, char_dim=self.args.char_dim,
                            hidden_dim=self.args.hidden_dim, num_layers=self.args.nlayers)
        elif self.args.model == 'mimick':
            model = EmbeddingModel(self.data, char_dim=self.args.char_dim, hidden_dim=self.args.hidden_dim)
        elif self.args.model == 'matrice':
            model = MatricesModel(self.data)
        else:
            model = None

        return model


def main():
    manager = EmbeddingTrainManager(args, parser)
    manager.train()


def old_main():
    data = EmbeddingData(args.batch_size)
    if args.model == 'myconv':
        model = MyConvModel(data, char_dim=args.char_dim, conv_channel=args.conv_channel)
    elif args.model == 'myrnn':
        model = MyModel(data, char_dim=args.char_dim, hidden_dim=args.hidden_dim, num_layers=args.nlayers)
    elif args.model == 'mimick':
        model = EmbeddingModel(data, char_dim=args.char_dim, hidden_dim=args.hidden_dim)
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
    fu.exist_make_dirs(log_dir)
    fu.exist_make_dirs(checkpoint_dir)

    loss = get_loss(args.loss, data, model) + get_loss(args.sec_loss, data, model)

    baseline = tf.losses.mean_squared_error(data.vec, model.output)

    global_step = tf.train.get_or_create_global_step()
    total_step = int(math.floor(data.num_example / args.batch_size))
    # learning_rate = tf.train.exponential_decay(args.learning_rate,
    #                                            global_step,
    #                                            args.decay_epoch * total_step,
    #                                            args.decay_rate,
    #                                            staircase=True)
    learning_rate = args.learning_rate
    optimizer = get_opt(args.optimizer, learning_rate)

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
    # global
    print(sys.argv)
    args, parser = args_parse()
    if args.mode == 'process':
        process()
    elif args.mode == 'tfrecord':
        create_records()
    elif args.mode == 'inspect':
        inspect()
    else:
        main()

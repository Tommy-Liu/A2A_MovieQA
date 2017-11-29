import argparse
import math
import os
import pprint
import time
from collections import Counter
from glob import glob
from multiprocessing import Pool
from os.path import join, exists
from random import shuffle

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tqdm import tqdm, trange

import data_utils as du
from config import MovieQAConfig
# from model import extract_axis_1
from qa_preprocessing import load_embedding

UNK = 'UNK'
RECORD_FILE_PATTERN = join('./embedding', 'embedding_dataset', 'embedding_%05d-of-%05d.tfrecord')
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
    RECORD_FILE_PATTERN_ = join('./embedding', 'embedding_dataset', 'embedding_*.tfrecord')

    def __init__(self, batch_size=128, num_thread=16):
        self.batch_size = batch_size
        self.num_example = len(np.load(config.encode_embedding_len_file))
        self.file_names = glob(self.RECORD_FILE_PATTERN_)
        self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
        self.dataset = tf.data.TFRecordDataset(self.file_names_placeholder) \
            .map(feature_parser, num_parallel_calls=num_thread).prefetch(num_thread * batch_size * 4) \
            .shuffle(buffer_size=num_thread * batch_size * 8).batch(batch_size)
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


class EmbeddingModel(object):
    def __init__(self, data, is_training=True, dropout_prob=1, char_embed_dim=100, hidden_dim=256):
        self.data = data
        if args.initializer == 'id':
            initializer = tf.identity_initializer()
        elif args.initializer == 'trunc':
            initializer = tf.truncated_normal_initializer(-1.0, 1.0)
        elif args.initializer == 'uni':
            initializer = tf.random_uniform_initializer(
                minval=-config.initializer_scale, maxval=config.initializer_scale)
        elif args.initializer == 'orth':
            initializer = tf.orthogonal_initializer()
        else:
            initializer = tf.glorot_uniform_initializer()

        embedding_matrix = tf.get_variable(
            name="embedding_matrix", initializer=initializer,
            shape=[self.data.vocab_size, char_embed_dim], trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)

        lstm_cell_fw = tf.nn.rnn_cell.GRUCell(hidden_dim,
                                              activation=tf.nn.relu,
                                              kernel_initializer=initializer,
                                              bias_initializer=tf.constant_initializer(0.1))

        lstm_cell_bw = tf.nn.rnn_cell.GRUCell(hidden_dim,
                                              activation=tf.nn.relu,
                                              kernel_initializer=initializer,
                                              bias_initializer=tf.constant_initializer(0.1))
        # init_fw_state = tf.get_variable('initial_forward_state', initializer=initializer,
        #                                 shape=[self.data.batch_size, hidden_dim], trainable=True)
        # init_bw_state = tf.get_variable('initial_backward_state', initializer=initializer,
        #                                 shape=[self.data.batch_size, hidden_dim], trainable=True)

        self.rnn_outputs, self.rnn_final_state = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, self.char_embedding, self.data.len, dtype=tf.float32)
        # init_fw_state, init_bw_state, tf.float32)
        # self.val_f, self.val_b = extract_axis_1(self.rnn_outputs[0],
        #                                         self.data.len - 1), \
        #                          extract_axis_1(self.rnn_outputs[1],
        #                                         np.zeros(2))
        # of, ob = o
        # of, ob0, ob1 = extract_axis_1(o[0], self.data.len - 1), extract_axis_1(o[1], np.zeros(64)), extract_axis_1(
        #     o[1], self.data.len - 1)
        # sf, sb = s
        self.hidden_state = tf.concat([self.rnn_final_state[0], self.rnn_final_state[1]], axis=1)
        self.lstm_to_wdim = layers.fully_connected(self.hidden_state, 300)
        # weights_initializer=initializer,
        # biases_initializer=tf.constant_initializer(0.1))
        self.output = layers.fully_connected(self.lstm_to_wdim, 300, activation_fn=None)
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
    if exists(config.w2v_embedding_key_file) and exists(config.w2v_embedding_vec_file):
        embedding_keys = du.load_json(config.w2v_embedding_key_file)
        embedding_vecs = np.load(config.w2v_embedding_vec_file)
    else:
        embedding = load_embedding(config.word2vec_file)
        embedding_keys = []
        embedding_vecs = np.zeros((len(embedding), embedding_size), dtype=np.float32)
        for i, k in enumerate(embedding.keys()):
            embedding_keys.append(k)
            embedding_vecs[i] = embedding[k]
        du.write_json(embedding_keys, config.w2v_embedding_key_file)
        np.save(config.w2v_embedding_vec_file, embedding_vecs)

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
    # U, s, V = np.linalg.svd(embedding_vecs)
    # print('Eigen value:')
    # pp.pprint(s)
    # print('Eigen vector:')
    # pp.pprint(V)
    vecs = embedding_vecs[list(keys.values())]
    embedding_keys, embedding_vecs = list(keys.keys()), vecs

    du.pprint(['Filtered number of embedding: %d' % len(embedding_keys),
               'Filtered shape of embedding vec: ' + str(embedding_vecs.shape),
               'Length\'s mean of keys: %.3f' % mean,
               'Length\'s std of keys: %.3f' % std,
               'Mean of embedding vecs: %.6f' % np.mean(embedding_vecs),
               'Std of embedding vecs: %.6f' % np.std(embedding_vecs),
               'Mean length of embedding vecs: %.6f' % np.mean(np.linalg.norm(embedding_vecs, axis=1)),
               'Std length of embedding vecs: %.6f' % np.std(np.linalg.norm(embedding_vecs, axis=1)),
               ])
    print('Element mean of embedding vec:')
    pp.pprint(np.mean(embedding_vecs, axis=0))


def process():
    # tokenize_qa = du.load_json(config.avail_tokenize_qa_file)
    # subtitle = du.load_json(config.subtitle_file)

    embedding_keys, embedding_vecs = load_embedding_vec()

    du.pprint(['w2v\'s # of embedding: %d' % len(embedding_keys),
               'w2v\'s shape of embedding vec: ' + str(embedding_vecs.shape)])

    filter_stat(embedding_keys, embedding_vecs)

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
    encode_embedding_keys = np.zeros((len(embedding_keys), args.max_length), dtype=np.int64)
    length = np.zeros(len(embedding_keys), dtype=np.int64)
    for i, k in enumerate(tqdm(embedding_keys, desc='Encoding...')):
        encode_embedding_keys[i, :len(k)] = [
            vocab.index(ch) if ch in vocab else vocab.index(UNK)
            for ch in k
        ]
        assert all([idx < len(vocab) for idx in encode_embedding_keys[i]]), \
            "Wrong index!"
        length[i] = len(k)
    du.pprint(['Shape of encoded key: %s' % encode_embedding_keys.shape,
               'Shape of encoded key length: %s' % length.shape])
    start_time = time.time()
    du.exist_then_remove(config.encode_embedding_key_file)
    du.exist_then_remove(config.encode_embedding_len_file)
    du.exist_then_remove(config.encode_embedding_vec_file)
    np.save(config.encode_embedding_key_file, encode_embedding_keys)
    np.save(config.encode_embedding_len_file, length)
    np.save(config.encode_embedding_vec_file, embedding_vecs)
    print('Saveing processed data with %.3f s' % (time.time() - start_time))


def instpect():
    # data = EmbeddingData(4096)
    # model = EmbeddingModel(data, dropout_prob=args.dropout_prob, char_embed_dim=args.char_embed_dim,
    #                        hidden_dim=args.hidden_dim, is_training=True)
    # norm_y, norm_y_ = tf.nn.l2_normalize(model.output, 1), tf.nn.l2_normalize(data.vec, 1)
    # loss = tf.losses.cosine_distance(norm_y_, norm_y, 1)
    # with tf.Session() as sess:
    #     sess.run(data.iterator.initializer, feed_dict={
    #         data.file_names_placeholder: data.file_names
    #     })
    #     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #     l, y, y_ = sess.run([loss, norm_y, norm_y_])
    #     print('Loss: %.4f' % l)
    #     print('Normalized output\'s shape:')
    #     pp.pprint(y.shape)
    #     print('Normalized label\'s shape:')
    #     pp.pprint(y_.shape)
    # final_state, f, b = sess.run([model.rnn_final_state, model.val_f, model.val_b])
    # print(np.array_equal(final_state[0], f))
    # print(np.array_equal(final_state[1], b))
    # print(final_state[0], '='*87, final_state[1], sep='\n')
    embedding_keys, embedding_vecs = load_embedding_vec()

    du.pprint(['w2v\'s # of embedding: %d' % len(embedding_keys),
               'w2v\'s shape of embedding vec: ' + str(embedding_vecs.shape)])

    filter_stat(embedding_keys, embedding_vecs)

    # vocab = du.load_json(config.char_vocab_file)
    # length = np.load(config.encode_embedding_len_file)
    # vecs = np.load(config.encode_embedding_vec_file)
    # lack = [ch for ch in string.ascii_lowercase + string.digits if ch not in vocab]
    #
    # print(lack)
    # print(vocab)
    # print(max(length))
    # print(vecs.shape)


def main():
    data = EmbeddingData(args.batch_size)
    model = EmbeddingModel(data, dropout_prob=args.dropout_prob, char_embed_dim=args.char_embed_dim,
                           hidden_dim=args.hidden_dim, is_training=True)
    log_dir = join(config.log_dir, 'embedding_log', '%.2E' % args.learning_rate)
    checkpoint_dir = join(config.checkpoint_dir, 'embedding_checkpoint', '%.2E' % args.learning_rate)
    checkpoint_name = join(checkpoint_dir, 'embedding')

    du.exist_make_dirs(log_dir)
    du.exist_make_dirs(checkpoint_dir)
    if args.reset:
        if exists(checkpoint_dir):
            os.system('rm -rf %s' % os.path.join(checkpoint_dir, '*'))
        if os.path.exists(log_dir):
            os.system('rm -rf %s' % os.path.join(log_dir, '*'))

    if args.loss == 'MSE':
        loss = tf.losses.mean_squared_error(data.vec, model.output)
    elif args.loss == 'ABS':
        loss = tf.losses.absolute_difference(data.vec, model.output)
    elif args.loss == 'L2':
        loss = tf.norm(data.vec - model.output)
    else:
        loss = tf.losses.cosine_distance(tf.nn.l2_normalize(data.vec, 1),
                                         tf.nn.l2_normalize(model.output, 1), 1)

    normalized_loss = tf.abs(tf.reduce_mean(model.output))
    loss += normalized_loss

    global_step = tf.train.get_or_create_global_step()
    total_step = int(math.floor(data.num_example / args.batch_size))
    learning_rate = tf.train.exponential_decay(args.learning_rate,
                                               global_step,
                                               args.decay_epoch * total_step,
                                               args.decay_rate,
                                               staircase=True)
    if args.optimizer == 'MOM':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif args.optimizer == 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    grads_and_vars = optimizer.compute_gradients(loss)
    gradients, variables = list(zip(*grads_and_vars))
    for var in variables:
        print(var.name, var.shape)
    capped_grads_and_vars = [(tf.clip_by_norm(gv[0], args.clip_norm), gv[1]) for gv in grads_and_vars]
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
        def train_loop(epoch_):
            shuffle(data.file_names)
            sess.run(data.iterator.initializer, feed_dict={
                data.file_names_placeholder: data.file_names,
            })
            step = tf.train.global_step(sess, global_step)
            print("Training Loop Epoch %d" % epoch_)
            step = step % total_step
            pbar = trange(step, total_step)
            for _ in pbar:
                try:
                    if step % total_step == total_step - 1:
                        _, l, n_l, step, y, y_ = sess.run(
                            [train_op, loss, normalized_loss, global_step, model.output, data.vec])
                        pp.pprint([y[-1], y_[-1]])
                        time.sleep(10)
                    elif step % 10 == 0:
                        gv_summary, summary, _, l, n_l, step = sess.run([train_gv_summaries_op, train_summaries_op,
                                                                         train_op, loss, normalized_loss, global_step])
                        save_sum(summary)
                        save_sum(gv_summary)
                        if step % 1000 == 0:
                            save()
                    else:
                        _, l, n_l, step = sess.run([train_op, loss, normalized_loss, global_step])
                    pbar.set_description(
                        '[%s/%s] step: %d total_loss: %.4f norm_loss: %.4f' % (epoch_, args.epoch, step, l, n_l))

                except tf.errors.OutOfRangeError:
                    break
                except KeyboardInterrupt:
                    save()
                    print()
                    return True

            print("Training Loop Epoch %d Done..." % epoch_)
            save()
            return False

        now_epoch = tf.train.global_step(sess, global_step) // total_step + 1
        for epoch in range(now_epoch, args.epoch + 1):
            if train_loop(epoch):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true',
                        help='Process the data which creating tfrecords needs.')
    parser.add_argument('--inspect', action='store_true', help='Inspect the data stat.')
    parser.add_argument('--num_shards', default=128, help='Number of tfrecords.', type=int)
    parser.add_argument('--tfrecord', action='store_true', help='Create tfrecords.')
    parser.add_argument('--checkpoint_file', default=None, help='Checkpoint file')
    parser.add_argument('--learning_rate', default=1E-6,
                        help='Initial learning rate.', type=float)
    parser.add_argument('--reset', action='store_true', help='Reset the experiment.')
    parser.add_argument('--batch_size', default=4096, help='Batch size of training.', type=int)
    parser.add_argument('--dropout_prob', default=1.0, help='Probability of dropout.', type=float)
    parser.add_argument('--char_embed_dim', default=100, help='Dimension of char embedding', type=int)
    parser.add_argument('--hidden_dim', default=128, help='Dimension of hidden state.', type=int)
    parser.add_argument('--epoch', default=200, help='Training epochs', type=int)
    parser.add_argument('--decay_epoch', default=10, help='Span of epochs at decay.', type=int)
    parser.add_argument('--decay_rate', default=0.87, help='Decay rate.', type=float)
    parser.add_argument('--optimizer', default='ADAM', help='Training policy.')
    parser.add_argument('--max_length', default=20, help='Maximal word length.', type=int)
    parser.add_argument('--loss', default='COS', help='Loss function')
    parser.add_argument('--clip_norm', default=0.1, help='Norm value of gradient clipping.', type=float)
    parser.add_argument('--initializer', default='id', help='Initializer of weight.')
    parser.add_argument('--multi_rnn', action='store_true', help='Multi-layer rnn.')

    args = parser.parse_args()
    print(vars(args))
    # if args.process:
    #     process()
    # if args.tfrecord:
    #     create_records()
    # if args.inspect:
    #     instpect()
    # if not (args.process or args.tfrecord or args.inspect):
    #     main()

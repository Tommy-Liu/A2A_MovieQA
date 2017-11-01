import argparse
import math
import os
import time
from collections import Counter
from glob import glob
from multiprocessing import Pool
from os.path import join, exists

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.data import TFRecordDataset
from tqdm import tqdm, trange

import data_utils as du
from config import MovieQAConfig
from qa_preprocessing import load_embedding

UNK = 'UNK'
RECORD_FILE_PATTERN = join('./data', 'dataset', 'embedding_%05d-of-%05d.tfrecord')

config = MovieQAConfig()


def feature_parser(record):
    features = {
        "vec": tf.FixedLenFeature([300], tf.float32),
        "word": tf.FixedLenFeature([98], tf.int64),
        "len": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, features)

    return parsed['vec'], parsed['word'], parsed['len']


class EmbeddingData(object):
    RECORD_FILE_PATTERN_ = join('./data', 'dataset', 'embedding_*.tfrecord')

    def __init__(self, batch_size=128, num_thread=4):
        self.num_example = len(np.load(config.encode_embedding_len_file))
        self.file_names = glob(self.RECORD_FILE_PATTERN_)
        self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
        self.dataset = TFRecordDataset(self.file_names_placeholder) \
            .map(feature_parser, num_threads=num_thread, output_buffer_size=num_thread * batch_size + 1000) \
            .shuffle(buffer_size=10000).batch(batch_size)
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
    def __init__(self, data, is_training=True):
        self.data = data
        embedding_matrix = tf.get_variable(
            name="embedding_matrix", shape=(self.data.vocab_size, 100), trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)
        initializer = tf.random_uniform_initializer(
            minval=-config.initializer_scale, maxval=config.initializer_scale)

        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(256, initializer=initializer)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell_fw, input_keep_prob=config.lstm_dropout_keep_prob,
            output_keep_prob=config.lstm_dropout_keep_prob) if is_training else lstm_cell_fw

        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(256, initializer=initializer)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell_bw, input_keep_prob=config.lstm_dropout_keep_prob,
            output_keep_prob=config.lstm_dropout_keep_prob) if is_training else lstm_cell_bw

        _, s = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, self.char_embedding, sequence_length=self.data.len, dtype=tf.float32)
        # of, ob = o
        # of, ob0, ob1 = extract_axis_1(o[0], self.data.len - 1), extract_axis_1(o[1], np.zeros(64)), extract_axis_1(
        #     o[1], self.data.len - 1)
        sf, sb = s
        self.hidden_state = tf.concat([sf.h, sb.h], axis=1)
        self.lstm_to_wdim = layers.dropout(layers.fully_connected(self.hidden_state, 300),
                                           keep_prob=config.lstm_dropout_keep_prob, is_training=is_training)
        self.output = layers.dropout(layers.fully_connected(self.lstm_to_wdim, 300),
                                     keep_prob=config.lstm_dropout_keep_prob, is_training=is_training)
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
    embedding_vec = np.load(config.w2v_embedding_npy_file)
    embedding_word = np.load(config.encode_embedding_file)
    embedding_word_length = np.load(config.encode_embedding_len_file)
    print('Loading file done. Spend %f sec' % (time.time() - start_time))
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


def process():
    tokenize_qa = du.load_json(config.avail_tokenize_qa_file)
    subtitle = du.load_json(config.subtitle_file)
    embedding = load_embedding(config.word2vec_file)

    embed_char_counter = Counter()
    for k in tqdm(embedding.keys()):
        embed_char_counter.update(k)

    embedding_keys = list(embedding.keys())
    embedding_array = np.array(list(embedding.values()), dtype=np.float32)
    du.write_json(embedding_keys, config.w2v_embedding_file)
    np.save(config.w2v_embedding_npy_file,
            embedding_array)

    qa_char_counter = Counter()
    for k in tokenize_qa.keys():
        for qa in tqdm(tokenize_qa[k], desc='Char counting %s' % k):
            for w in qa['tokenize_question']:
                qa_char_counter.update(w)
            for a in qa['tokenize_answer']:
                for w in a:
                    qa_char_counter.update(w)
            for v in qa['video_clips']:
                for l in subtitle[v]:
                    for w in l:
                        qa_char_counter.update(w)

    du.write_json(embed_char_counter, config.embed_char_counter_file)
    du.write_json(qa_char_counter, config.qa_char_counter_file)

    count_array = np.array(list(embed_char_counter.values()), dtype=np.float32)
    m, v, md, f = np.mean(count_array), np.std(count_array), np.median(count_array), np.percentile(count_array, 95)
    print(m, v, md, f)

    above_mean = dict(filter(lambda item: item[1] > f, embed_char_counter.items()))
    below_mean = dict(filter(lambda item: item[1] < f, embed_char_counter.items()))
    below_occur = set(filter(lambda k: k in qa_char_counter, below_mean.keys()))
    final_set = below_occur.union(set(above_mean.keys()))
    du.write_json(list(final_set) + [UNK], config.char_vocab_file)

    vocab = du.load_json(config.char_vocab_file)
    encode_embedding_keys = np.zeros((len(embedding_keys), 98), dtype=np.int64)
    length = np.zeros(len(embedding_keys), dtype=np.int64)
    for i, k in enumerate(tqdm(embedding_keys, desc='OP')):
        encode_embedding_keys[i, :len(k)] = [
            vocab.index(ch) if ch in vocab else vocab.index('UNK')
            for ch in k
        ]
        length[i] = len(k)

    np.save(config.encode_embedding_file, encode_embedding_keys)
    np.save(config.encode_embedding_len_file, length)


def main():
    data = EmbeddingData(args.batch_size)
    model = EmbeddingModel(data, is_training=False)
    log_dir = join(config.log_dir, 'embedding_log', '%.2E' % args.initial_learning_rate)
    checkpoint_dir = join(config.checkpoint_dir, 'embedding_checkpoint', '%.2E' % args.initial_learning_rate)
    checkpoint_name = join(checkpoint_dir, 'embedding')

    du.exist_make_dirs(log_dir)
    du.exist_make_dirs(checkpoint_dir)
    if args.reset:
        if exists(checkpoint_dir):
            os.system('rm -rf %s' % os.path.join(checkpoint_dir, '*'))
        if os.path.exists(log_dir):
            os.system('rm -rf %s' % os.path.join(log_dir, '*'))

    loss = tf.norm(data.vec - model.output)
    global_step = tf.train.get_or_create_global_step()
    total_step = int(math.floor(data.num_example / args.batch_size))
    learning_rate = tf.train.exponential_decay(args.initial_learning_rate,
                                               global_step,
                                               config.num_epochs_per_decay * total_step,
                                               config.learning_rate_decay_factor,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    gradients, variables = list(zip(*grads_and_vars))
    gradients, _ = tf.clip_by_global_norm(gradients, config.clip_gradients)
    capped_grad_and_vars = list(zip(gradients, variables))
    train_op = optimizer.apply_gradients(capped_grad_and_vars, global_step)
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

    if args.checkpoint_file:
        checkpoint_file = args.checkpoint_file
    else:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    restore_fn = (lambda _sess: saver.restore(_sess, checkpoint_file)) \
        if checkpoint_file else None

    sv = tf.train.Supervisor(logdir=log_dir, summary_op=None,
                             init_fn=restore_fn, save_model_secs=0,
                             saver=saver, global_step=global_step)

    config_ = tf.ConfigProto(allow_soft_placement=True, )
    # config_.gpu_options.allow_growth = True

    with sv.managed_session(config=config_) as sess:
        def save():
            sv.saver.save(sess, checkpoint_name, tf.train.global_step(sess, global_step))

        def save_sum(summary_):
            sv.summary_computed(sess, summary_, tf.train.global_step(sess, global_step))

        # Training loop
        def train_loop(epoch_):
            sess.run(data.iterator.initializer, feed_dict={
                data.file_names_placeholder: data.file_names,
            })
            step = tf.train.global_step(sess, global_step)
            print("Training Loop Epoch %d" % epoch_)
            step = step % total_step

            for _ in range(step, total_step):
                start_time = time.time()
                try:
                    if step % 1000 == 0:
                        gv_summary, summary, _, l, step = sess.run([train_gv_summaries_op, train_summaries_op,
                                                                    train_op, loss, global_step])
                        save_sum(summary)
                        save_sum(gv_summary)
                        save()
                    elif step % 10 == 0:
                        summary, _, l, step = sess.run([train_summaries_op, train_op, loss, global_step])
                        save_sum(summary)
                    else:
                        _, l, step = sess.run([train_op, loss, global_step])
                    print("[%s/%s] step: %d loss: %.3f elapsed time: %.2f s" %
                          (epoch_, config.num_epochs, step, l, time.time() - start_time))

                except tf.errors.OutOfRangeError:
                    break
                except KeyboardInterrupt:
                    save()
                    print()
                    return True
            print("Training Loop Epoch %d Done..." % epoch_)
            save()
            return False

        now_epoch = tf.train.global_step(sess, global_step) // data.num_example + 1
        for epoch in range(now_epoch, config.num_epochs + 1):
            if train_loop(epoch):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true',
                        help='Process the data which creating tfrecords needs.')
    parser.add_argument('--num_shards', default=128, help='Number of tfrecords.', type=int)
    parser.add_argument('--tfrecord', action='store_true', help='Create tfrecords.')
    parser.add_argument('--checkpoint_file', default=None, help='Checkpoint file')
    parser.add_argument('--initial_learning_rate', default=config.initial_learning_rate,
                        help='Initial learning rate.', type=float)
    parser.add_argument('--reset', action='store_true', help='Reset the experiment.')
    parser.add_argument('--batch_size', default=3072, help='Batch size of training.', type=int)
    args = parser.parse_args()
    if args.process:
        process()
    if args.tfrecord:
        create_records()

    main()

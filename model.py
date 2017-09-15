import pprint

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as l
import tensorflow.contrib.slim as slim

from config import MovieQAConfig
from get_dataset import MovieQAData

time_steps = 20
frame_size = 20000


def npy_read_func(file_name):
    return np.load(file_name.decode('utf-8'))


class VLLabMemoryModel(object):
    def __init__(self, data, config, is_training=True):
        self.data = data
        self.config = config
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self.ques_embeddings = None
        self.ans_embeddings = None
        self.subt_embedding = None
        self.mean_subt = None

        self.ques_lstm_outputs = None
        self.ans_lstm_outputs = None

        self.movie_feature_repr = None
        self.movie_convpool = None
        self.final_repr = None

        self.logits = None
        self.prediction = None
        # self.subt_lstm_outputs = None

        self.build_model()

        # self.simple_input_pipeline()

    def sanity_check(self, sess):
        pprint.pprint(self.__dict__)
        # [ins if ins for ins in self.__dict__.values()]
        # sess.run()

    def build_model(self):
        self.build_seq_embedding()
        self.build_movie_feature()
        self.build_sliding_conv()
        self.multilayer_perceptron()

    def build_movie_feature(self):
        with tf.variable_scope('MovieRepr'):
            concat_feature_repr = tf.concat([self.data.feat, self.mean_subt], axis=1)
            concat_feature_repr = tf.stack([concat_feature_repr for _ in range(self.config.batch_size)])
            self.movie_feature_repr = tf.expand_dims(concat_feature_repr, axis=3)

    def multilayer_perceptron(self):
        with tf.variable_scope("MLP"):
            x = l.dropout(self.final_repr, keep_prob=self.config.lstm_dropout_keep_prob)
            fc1 = l.fully_connected(x, 2048)
            fc1 = l.dropout(fc1, keep_prob=self.config.lstm_dropout_keep_prob)
            fc2 = l.fully_connected(fc1, 1024)
            fc2 = l.dropout(fc2, keep_prob=self.config.lstm_dropout_keep_prob)
            fc3 = l.fully_connected(fc2, 512)
            fc3 = l.dropout(fc3, keep_prob=self.config.lstm_dropout_keep_prob)
            self.logits = l.fully_connected(fc3, 1, activation_fn=None)
            self.prediction = tf.nn.sigmoid(self.logits)

    def build_seq_embedding(self):
        with tf.variable_scope("SeqEmbedding"):
            with tf.variable_scope("QuesEmbedding"):
                with tf.device("/cpu:0"):
                    embedding_map = tf.get_variable(
                        name="map",
                        shape=[self.config.size_vocab_q, self.config.embedding_size],
                        initializer=self.initializer)
                    seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.data.ques)
                    self.ques_embeddings = seq_embeddings
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_lstm_units,
                                                    initializer=self.initializer)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                          input_keep_prob=self.config.lstm_dropout_keep_prob,
                                                          output_keep_prob=self.config.lstm_dropout_keep_prob)
                zero_state = lstm_cell.zero_state(
                    batch_size=self.config.batch_size, dtype=tf.float32)

                _, self.ques_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                              inputs=self.ques_embeddings,
                                                              sequence_length=self.data.ques_length,
                                                              initial_state=zero_state,
                                                              dtype=tf.float32)

            with tf.variable_scope("AnsEmbedding"):
                with tf.device("/cpu:0"):
                    embedding_map = tf.get_variable(
                        name="map",
                        shape=[self.config.size_vocab_a, self.config.embedding_size],
                        initializer=self.initializer)
                    seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.data.ans)
                    self.ans_embeddings = seq_embeddings
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_lstm_units,
                                                    initializer=self.initializer)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                          input_keep_prob=self.config.lstm_dropout_keep_prob,
                                                          output_keep_prob=self.config.lstm_dropout_keep_prob)
                zero_state = lstm_cell.zero_state(
                    batch_size=self.config.batch_size, dtype=tf.float32)

                _, self.ans_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                             inputs=self.ans_embeddings,
                                                             sequence_length=self.data.ans_length,
                                                             initial_state=zero_state,
                                                             dtype=tf.float32)

            with tf.variable_scope("SubtEmbedding"):
                with tf.device("/cpu:0"):
                    embedding_map = tf.get_variable(
                        name="map",
                        shape=[self.config.size_vocab_s, self.config.embedding_size],
                        initializer=self.initializer)
                    seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.data.subt)
                    self.subt_embedding = seq_embeddings
                subt_mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.data.subt_length), axis=2),
                                    [1, 1, self.config.embedding_size])
                zeros = tf.zeros_like(self.subt_embedding)
                masked_subt = tf.where(subt_mask, self.subt_embedding, zeros)

                self.mean_subt = tf.divide(tf.reduce_sum(masked_subt, axis=1),
                                           tf.expand_dims(tf.cast(self.data.subt_length, tf.float32), axis=1))
                # _, self.subt_partition = tf.dynamic_partition(self.subt_embedding, subt_mask,
                #                                               num_partitions=2)
                # lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_lstm_units,
                #                                     initializer=self.initializer)
                # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                #                                           input_keep_prob=self.config.lstm_dropout_keep_prob,
                #                                           output_keep_prob=self.config.lstm_dropout_keep_prob)
                # zero_state = lstm_cell.zero_state(
                #     batch_size=self.config.batch_size, dtype=tf.float32)
                #
                # _, self.subt_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                #                                               inputs=self.subt_embedding,
                #                                               sequence_length=self.data.subt_length,
                #                                               initial_state=zero_state,
                #                                               dtype=tf.float32)

    def build_sliding_conv(self, layers=1):
        x = [self.movie_feature_repr]
        input_channel = self.config.feature_dim + self.config.embedding_size
        conv_outputs = []
        for layer in range(1, layers + 1):
            conv_outputs = []
            output_channel = self.config.sliding_dim * (2 ** (layers - layer))
            with tf.variable_scope("SlideConv-%s" % layer):
                for inp in x:
                    for filter_size in self.config.filter_sizes:
                        with tf.variable_scope("Conv-%s" % filter_size):
                            conv = l.conv2d(inp, output_channel,
                                            [filter_size, input_channel],
                                            padding='VALID')
                            conv = l.dropout(conv, self.config.lstm_dropout_keep_prob)
                            conv_outputs.append(conv)
            x = conv_outputs
        with tf.variable_scope("Avgpool"):
            pooled_outputs = []
            for conv in conv_outputs:
                # Max-pooling over the outputs
                pooled = tf.reduce_mean(conv, axis=1, keep_dims=True)
                pooled_outputs.append(pooled)

            concat_pool = tf.concat(pooled_outputs, axis=3)
            concat_pool = tf.squeeze(concat_pool)
        self.movie_convpool = concat_pool
        self.final_repr = tf.concat([self.movie_convpool, self.ans_lstm_outputs.h, self.ques_lstm_outputs.h], axis=1)


def main(_):
    config_ = MovieQAConfig()
    data = MovieQAData(config_)
    model = VLLabMemoryModel(data, config_)
    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    # print('Start extract !!')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(data.iterator.initializer, feed_dict={
            data.file_names_placeholder: data.file_names
        })
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        i = 1
        try:
            while True:
                l, p = sess.run([model.logits, model.prediction])
                print(l, p, sep='\n')
                print('At [%5d/%5d]' % (i, config_.num_training_train_examples))
                i += 1
        except tf.errors.OutOfRangeError:
            print('Done!')
        except KeyboardInterrupt:
            print()
        finally:
            print(i)
            # coord.request_stop()
            # coord.join(threads)
            # qe, ae, se = sess.run([model.ques_lstm_outputs,
            #                        model.ans_lstm_outputs,
            #                        model.mean_subt])
            #
            # print(qe.h)
            # print(ae.h)
            # print(se)
            # print(qe.h.shape, ae.h.shape, se.shape)
            # pooled = sess.run(model.final_repr)
            # print(pooled)
            # print(pooled.shape)
            # for pool in pooled:
            #     print(pool)
            #     print(pool.shape)


if __name__ == '__main__':
    tf.app.run()

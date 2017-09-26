import pprint

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from config import MovieQAConfig
from get_dataset import MovieQAData

time_steps = 20
frame_size = 20000

config = MovieQAConfig()


def npy_read_func(file_name):
    return np.load(file_name.decode('utf-8'))


def extract_axis_1(data, ind):
    batch_range = tf.cast(tf.range(tf.shape(data)[0]), dtype=tf.int64)
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


class VLLabMemoryModel(object):
    def __init__(self, data, num_layers=1, is_training=True):
        if is_training:
            self.batch_size = 2
        else:
            self.batch_size = 5
        self.data = data
        self.initializer = tf.random_uniform_initializer(
            minval=-config.initializer_scale,
            maxval=config.initializer_scale)

        self.ques_embeddings = None
        self.ans_embeddings = None
        self.subt_embeddings = None
        self.mean_subt = None

        self.ques_lstm_outputs = None
        self.ans_lstm_outputs = None
        self.subt_lstm_outputs = None

        self.ques_lstm_final_state = None
        self.ans_lstm_final_state = None
        self.subt_lstm_final_state = None

        self.movie_repr = None
        self.movie_convpool = None
        self.final_repr = None

        self.logits = None
        self.prediction = None
        self.num_layers = num_layers

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
        with tf.variable_scope('movie_repr'):
            concat_feature_repr = tf.concat([self.data.feat, self.subt_lstm_outputs], axis=1)
            concat_feature_repr = tf.stack([concat_feature_repr for _ in range(self.batch_size)])
            self.movie_repr = tf.expand_dims(concat_feature_repr, axis=3)

    def multilayer_perceptron(self):
        with tf.variable_scope("mlp"):
            x = layers.dropout(self.final_repr, keep_prob=config.lstm_dropout_keep_prob)
            # x = self.final_repr
            fc1 = layers.fully_connected(x, 2048)
            fc1 = layers.dropout(fc1, keep_prob=config.lstm_dropout_keep_prob)
            fc2 = layers.fully_connected(fc1, 1024)
            fc2 = layers.dropout(fc2, keep_prob=config.lstm_dropout_keep_prob)
            fc3 = layers.fully_connected(fc2, 512)
            fc3 = layers.dropout(fc3, keep_prob=config.lstm_dropout_keep_prob)
            # fc3 = x
            self.logits = layers.fully_connected(fc3, 1, activation_fn=None)
            self.prediction = tf.nn.sigmoid(self.logits)

    def lstm(self, x, length):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(config.num_lstm_units, initializer=self.initializer, )
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                  input_keep_prob=config.lstm_dropout_keep_prob,
                                                  output_keep_prob=config.lstm_dropout_keep_prob)
        o, s = tf.nn.dynamic_rnn(cell=lstm_cell,
                                 inputs=x,
                                 sequence_length=length,
                                 dtype=tf.float32)

        o = extract_axis_1(o, length - 1)
        return o, s

    def mean_embedding(self, x, length, max_length):
        subt_mask = tf.tile(
            tf.expand_dims(
                tf.sequence_mask(length,
                                 maxlen=max_length), axis=2),
            [1, 1, config.embedding_size])
        zeros = tf.zeros_like(x)
        masked_x = tf.where(subt_mask, x, zeros)

        return tf.divide(tf.reduce_sum(masked_x, axis=1),
                         tf.expand_dims(tf.cast(length, tf.float32), axis=1))

    def build_seq_embedding(self):
        with tf.variable_scope("seq_embedding"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[config.size_vocab, config.embedding_size],
                initializer=self.initializer)
            self.ques_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.data.ques)
            self.ans_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.data.ans)
            self.subt_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.data.subt)

        # self.ques_lstm_outputs = self.mean_embedding(
        #     self.ques_embeddings, self.data.ques_length, config.ques_max_length)
        # self.ans_lstm_outputs = self.mean_embedding(
        #     self.ans_embeddings, self.data.ans_length, config.ans_max_length)
        # self.subt_lstm_outputs = self.mean_embedding(
        #     self.subt_embeddings, self.data.subt_length, config.subt_max_length)

        with tf.variable_scope("ques_lstm"):
            self.ques_lstm_outputs, self.ques_lstm_final_state = \
                self.lstm(self.ques_embeddings, self.data.ques_length)

        with tf.variable_scope("ans_lstm"):
            self.ans_lstm_outputs, self.ans_lstm_final_state = \
                self.lstm(self.ans_embeddings, self.data.ans_length)

        with tf.variable_scope("subt_lstm"):
            self.subt_lstm_outputs, self.subt_lstm_final_state = \
                self.lstm(self.subt_embeddings, self.data.subt_length)

    def build_sliding_conv(self):
        x = [self.movie_repr]
        print(self.movie_repr.shape)
        conv_outputs = []
        width = config.feature_dim + config.num_lstm_units
        for layer in range(1, self.num_layers + 1):
            conv_outputs = []
            output_channel = config.sliding_dim * (2 ** (self.num_layers - layer))

            with tf.variable_scope("slide_conv_%s" % layer):
                for inp in x:
                    for filter_size in config.filter_sizes:
                        with tf.variable_scope("conv_%s" % filter_size):
                            conv = layers.conv2d(inp, output_channel,
                                                 [filter_size, width],
                                                 padding='VALID')
                            conv = layers.dropout(conv, config.lstm_dropout_keep_prob)
                            conv = tf.transpose(conv, perm=[0, 1, 3, 2])
                            print(conv.shape)
                            conv_outputs.append(conv)
            width = output_channel
            x = conv_outputs
        with tf.variable_scope("avgpool"):
            pooled_outputs = []
            for conv in conv_outputs:
                # Mean-pooling over the outputs
                pooled = tf.reduce_mean(conv, axis=1)
                print(pooled.shape)
                pooled_outputs.append(pooled)

            concat_pool = tf.concat(pooled_outputs, axis=1)
            concat_pool = tf.squeeze(concat_pool)
        self.movie_convpool = concat_pool
        self.final_repr = tf.concat([self.movie_convpool,
                                     self.ques_lstm_outputs, self.ans_lstm_outputs],
                                    axis=1)


def main(_):
    data = MovieQAData()
    model = VLLabMemoryModel(data)
    config_ = tf.ConfigProto(allow_soft_placement=True, )
    config_.gpu_options.allow_growth = True
    # print('Start extract !!')
    with tf.Session(config=config_) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(data.iterator.initializer, feed_dict={
            data.file_names_placeholder: data.file_names
        })
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        try:
            while True:
                loss, p = sess.run([model.logits, model.prediction])
                print(loss, p, sep='\n')
                print('At [%5d/%5d]' % (i, config_.get_num_example(config_.dataset_name, )))
                i += 1
        except tf.errors.OutOfRangeError:
            print('Done!')
        except KeyboardInterrupt:
            print()
        finally:
            print(i, config_.get_num_example(config_.dataset_name, ))
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

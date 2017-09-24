import pprint

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from config import MovieQAConfig
from get_dataset import MovieQAData

time_steps = 20
frame_size = 20000


def npy_read_func(file_name):
    return np.load(file_name.decode('utf-8'))


class VLLabMemoryModel(MovieQAConfig):
    def __init__(self, data, num_layers=1, is_training=True):
        super(VLLabMemoryModel, self).__init__()
        self.data = data
        self.initializer = tf.random_uniform_initializer(
            minval=-self.initializer_scale,
            maxval=self.initializer_scale)

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
        with tf.variable_scope('MovieRepr'):
            concat_feature_repr = tf.concat([self.data.feat, self.mean_subt], axis=1)
            concat_feature_repr = tf.stack([concat_feature_repr for _ in range(self.batch_size)])
            self.movie_feature_repr = tf.expand_dims(concat_feature_repr, axis=3)

    def multilayer_perceptron(self):
        with tf.variable_scope("MLP"):
            x = layers.dropout(self.final_repr, keep_prob=self.lstm_dropout_keep_prob)
            fc1 = layers.fully_connected(x, 2048)
            fc1 = layers.dropout(fc1, keep_prob=self.lstm_dropout_keep_prob)
            fc2 = layers.fully_connected(fc1, 1024)
            fc2 = layers.dropout(fc2, keep_prob=self.lstm_dropout_keep_prob)
            fc3 = layers.fully_connected(fc2, 512)
            fc3 = layers.dropout(fc3, keep_prob=self.lstm_dropout_keep_prob)
            self.logits = layers.fully_connected(fc3, 1, activation_fn=None)
            self.prediction = tf.nn.sigmoid(self.logits)

    def build_seq_embedding(self):
        with tf.variable_scope("SeqEmbedding"):
            with tf.device("/cpu:0"):
                embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.size_vocab, self.embedding_size],
                    initializer=self.initializer)
            with tf.variable_scope("QuesEmbedding"):
                self.ques_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.data.ques)
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units,
                                                    initializer=self.initializer)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                          input_keep_prob=self.lstm_dropout_keep_prob,
                                                          output_keep_prob=self.lstm_dropout_keep_prob)
                zero_state = lstm_cell.zero_state(
                    batch_size=self.batch_size, dtype=tf.float32)

                _, self.ques_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                              inputs=self.ques_embeddings,
                                                              sequence_length=self.data.ques_length,
                                                              initial_state=zero_state,
                                                              dtype=tf.float32)

            with tf.variable_scope("AnsEmbedding"):
                self.ans_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.data.ans)
                lstm_cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units,
                                                    initializer=self.initializer)
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                          input_keep_prob=self.lstm_dropout_keep_prob,
                                                          output_keep_prob=self.lstm_dropout_keep_prob)
                zero_state = lstm_cell.zero_state(
                    batch_size=self.batch_size, dtype=tf.float32)

                _, self.ans_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                             inputs=self.ans_embeddings,
                                                             sequence_length=self.data.ans_length,
                                                             initial_state=zero_state,
                                                             dtype=tf.float32)

            with tf.variable_scope("SubtEmbedding"):
                self.subt_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.subt)
                subt_mask = tf.tile(
                    tf.expand_dims(
                        tf.sequence_mask(self.data.subt_length,
                                         maxlen=self.subt_max_length), axis=2),
                    [1, 1, self.embedding_size])
                zeros = tf.zeros_like(self.subt_embedding)
                masked_subt = tf.where(subt_mask, self.subt_embedding, zeros)

                self.mean_subt = tf.divide(tf.reduce_sum(masked_subt, axis=1),
                                           tf.expand_dims(tf.cast(self.data.subt_length, tf.float32), axis=1))

    def build_sliding_conv(self):
        x = [self.movie_feature_repr]
        input_channel = self.feature_dim + self.embedding_size
        conv_outputs = []
        for layer in range(1, self.num_layers + 1):
            conv_outputs = []
            output_channel = self.sliding_dim * (2 ** (self.num_layers - layer))
            with tf.variable_scope("SlideConv-%s" % layer):
                for inp in x:
                    for filter_size in self.filter_sizes:
                        with tf.variable_scope("Conv-%s" % filter_size):
                            conv = layers.conv2d(inp, output_channel,
                                                 [filter_size, input_channel],
                                                 padding='VALID')
                            conv = layers.dropout(conv, self.lstm_dropout_keep_prob)
                            conv_outputs.append(conv)
            x = conv_outputs
        with tf.variable_scope("Avgpool"):
            pooled_outputs = []
            for conv in conv_outputs:
                # Mean-pooling over the outputs
                pooled = tf.reduce_mean(conv, axis=1, keep_dims=True)
                pooled_outputs.append(pooled)

            concat_pool = tf.concat(pooled_outputs, axis=3)
            concat_pool = tf.squeeze(concat_pool)
        self.movie_convpool = concat_pool
        self.final_repr = tf.concat([self.movie_convpool, self.ans_lstm_outputs.h, self.ques_lstm_outputs.h], axis=1)


def main(_):
    config_ = MovieQAConfig()
    data = MovieQAData(config_)
    model = VLLabMemoryModel(data)
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

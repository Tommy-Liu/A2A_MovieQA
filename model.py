import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as l

from config import MovieQAConfig
from get_dataset import MovieQAData

time_steps = 20
frame_size = 20000


def npy_read_func(file_name):
    return np.load(file_name.decode('utf-8'))


class VLLabMemoryModel(object):
    def __init__(self, data, is_training=True):
        self.data = data
        self.config = MovieQAConfig()
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self.ques_embeddings = None
        self.ans_embeddings = None
        self.subt_embedding = None

        self.ques_lstm_outputs = None
        self.ans_lstm_outputs = None

        self.logits = None
        self.prediction = None
        # self.subt_lstm_outputs = None


        # self.seq_index = tf.convert_to_tensor(
        #     np.random.randint(1000, size=(self.config.batch_size, time_steps)), dtype=tf.int64)
        self.batch_features = None
        # self.batch_features = tf.convert_to_tensor(
        #     np.random.randint(1000, size=(self.config.batch_size, frame_size, self.config.feature_dim, 1)),
        #     dtype=tf.float32)
        # self.mask = tf.convert_to_tensor(np.array([
        #     [1] * i * 2 + [0] * (time_steps - i * 2) for i in range(1, self.config.batch_size + 1)
        # ]), dtype=tf.int64)

        self.build_seq_embedding()
        concat_feature_repr = tf.concat([self.data.feat, self.mean_subt], axis=1)
        concat_feature_repr = tf.stack([concat_feature_repr for _ in range(self.config.batch_size)])
        self.concat_feature_repr = tf.expand_dims(concat_feature_repr, axis=3)
        self.concat_pool = self.sliding_conv(self.concat_feature_repr)
        self.final_repr = tf.concat([self.concat_pool, self.ans_lstm_outputs.h, self.ques_lstm_outputs.h], axis=1)
        self.multilayer_perceptron()
        # self.simple_input_pipeline()
        # self.sliding_conv()

    def multilayer_perceptron(self):
        with tf.variable_scope("MLP"):
            fc1 = l.fully_connected(self.final_repr, 2048)
            fc1 = l.dropout(fc1, keep_prob=self.config.lstm_dropout_keep_prob)
            fc2 = l.fully_connected(fc1, 1024)
            fc2 = l.dropout(fc2, keep_prob=self.config.lstm_dropout_keep_prob)
            fc3 = l.fully_connected(fc2, 512)
            fc3 = l.dropout(fc3, keep_prob=self.config.lstm_dropout_keep_prob)
            self.logits = l.fully_connected(fc3, 1, activation_fn=None)
            self.prediction = tf.nn.sigmoid(self.logits)

    def build_seq_embedding(self):
        with tf.variable_scope("ques_embedding"):
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

        with tf.variable_scope("ans_embedding"):
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

        with tf.variable_scope("subt_embedding"):
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

    def sliding_conv(self, x):
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # filter_shape = [filter_size, self.config.feature_dim+self.config.embedding_size, 1, self.config.sliding_dim]
                # weight = tf.get_variable("weight", shape=filter_shape,
                #                          initializer=l.xavier_initializer(),
                #                          )
                # bias = tf.get_variable("bias", shape=[self.config.sliding_dim],
                #                        initializer=tf.constant_initializer(0.0))
                # conv = tf.nn.conv2d(x,
                #                     weight, [1, 1, 1, 1],
                #                     'VALID', name="conv")
                # conv = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
                conv = l.conv2d(x, self.config.sliding_dim,
                                [filter_size, self.config.feature_dim + self.config.embedding_size],
                                padding='VALID')
                conv = l.dropout(conv, self.config.lstm_dropout_keep_prob)
                # Max-pooling over the outputs
                pooled = tf.reduce_mean(conv, axis=1, keep_dims=True)
                pooled_outputs.append(pooled)
        concat_pool = tf.concat(pooled_outputs, axis=3)
        concat_pool = tf.squeeze(concat_pool)
        return concat_pool

    def simple_input_pipeline(self):
        # print(self.config.npy_files)
        filename_queue = tf.train.string_input_producer(self.config.npy_files, shuffle=False)
        reader = tf.IdentityReader()
        _, filename_tensor = reader.read(filename_queue)
        feature = tf.py_func(npy_read_func, [filename_tensor], tf.float32)
        feature = tf.reshape(feature, [-1, self.config.feature_dim, 1])
        features = tf.train.batch([feature], batch_size=self.config.batch_size,
                                  num_threads=self.config.num_worker,
                                  dynamic_pad=True,
                                  allow_smaller_final_batch=True)
        self.batch_features = features


def main(_):
    data = MovieQAData()
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
        i = 1
        try:
            while True:
                l, p = sess.run([model.logits, model.prediction])
                print(l, p, sep='\n')
                print('At [%5d/%5d]' % (i, data.num_samples))
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


def test():
    batch_features = tf.convert_to_tensor(np.random.rand(4, time_steps, 1536), dtype=tf.float32)

    mask = tf.convert_to_tensor(np.array([
        [1] * i + [0] * (time_steps - i) for i in range(1, 4 + 1)
    ]), dtype=tf.int64)

    initializer = tf.random_uniform_initializer(
        minval=-0.08,
        maxval=0.08)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(512, initializer=initializer)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                              input_keep_prob=0.7,
                                              output_keep_prob=0.7)

    zero_state = lstm_cell.zero_state(
        batch_size=4, dtype=tf.float32)

    sequence_length = tf.reduce_sum(mask, 1)
    lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                        inputs=batch_features,
                                        sequence_length=sequence_length,
                                        initial_state=zero_state,
                                        dtype=tf.float32)

    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    print('Start extract !!')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        l_o = sess.run(lstm_outputs)
        print(l_o[:, :4, :].shape)
        print(l_o[:, :4, :])


if __name__ == '__main__':
    tf.app.run()

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as l

from config import ModelConfig

time_steps = 20
frame_size = 20000


def npy_read_func(file_name):
    return np.load(file_name.decode('utf-8'))


class VLLabMemoryModel(object):
    def __init__(self, inputs):

        self.config = ModelConfig()
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self.ques_embeddings = None
        self.ans_embeddings = None
        self.subt_embedding = None

        self.ques_lstm_outputs = None
        self.ans_lstm_outputs = None
        self.subt_lstm_outputs = None

        self.conv_test = []
        self.pooled_outputs = []

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
        # self.simple_input_pipeline()
        # self.sliding_conv()

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

    def sliding_conv(self):
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.feature_dim, 1, self.config.sliding_dim]
                weight = tf.get_variable("weight", shape=filter_shape,
                                         initializer=l.xavier_initializer(),
                                         )
                bias = tf.get_variable("bias", shape=[self.config.sliding_dim],
                                       initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(self.batch_features,
                                    weight, [1, 1, 1, 1],
                                    'VALID', name="conv")
                conv = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
                # conv = tf.transpose(conv, perm=[0, 1, 3, 2])
                self.conv_test.append(conv)
                # Max-pooling over the outputs
                pooled = tf.reduce_max(conv, axis=1, keep_dims=True)
                # pooled = tf.nn.max_pool(
                #     conv,
                #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                self.pooled_outputs.append(pooled)

                # Combine all the pooled features
                # num_filters_total = num_filters * len(filter_sizes)
                # self.h_pool = tf.concat(3, pooled_outputs)
                # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

    def build_seq_embedding(self):
        with tf.variable_scope("ques_embedding"):
            with tf.device("/cpu:0"):
                embedding_map = tf.get_variable(
                    name="map",
                    shape=[self.config.size_vocab_q, self.config.embedding_size],
                    initializer=self.initializer)
                seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.seq_index)
                self.ques_embeddings = seq_embeddings
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_lstm_units,
                                                initializer=self.initializer)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      input_keep_prob=self.config.lstm_dropout_keep_prob,
                                                      output_keep_prob=self.config.lstm_dropout_keep_prob)
            zero_state = lstm_cell.zero_state(
                batch_size=self.config.batch_size, dtype=tf.float32)

            sequence_length = tf.reduce_sum(self.mask, 1)
            _, self.ques_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                          inputs=self.ques_embeddings,
                                                          sequence_length=sequence_length,
                                                          initial_state=zero_state,
                                                          dtype=tf.float32)

        with tf.variable_scope("ans_embedding"):
            with tf.device("/cpu:0"):
                embedding_map = tf.get_variable(
                    name="map",
                    shape=[self.config.size_vocab_q, self.config.embedding_size],
                    initializer=self.initializer)
                seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.seq_index)
                self.ans_embeddings = seq_embeddings
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_lstm_units,
                                                initializer=self.initializer)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      input_keep_prob=self.config.lstm_dropout_keep_prob,
                                                      output_keep_prob=self.config.lstm_dropout_keep_prob)
            zero_state = lstm_cell.zero_state(
                batch_size=self.config.batch_size, dtype=tf.float32)

            sequence_length = tf.reduce_sum(self.mask, 1)
            _, self.ans_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                         inputs=self.ans_embeddings,
                                                         sequence_length=sequence_length,
                                                         initial_state=zero_state,
                                                         dtype=tf.float32)

        with tf.variable_scope("subt_embedding"):
            with tf.device("/cpu:0"):
                embedding_map = tf.get_variable(
                    name="map",
                    shape=[self.config.size_vocab_q, self.config.embedding_size],
                    initializer=self.initializer)
                seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.seq_index)
                self.subt_embedding = seq_embeddings
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.num_lstm_units,
                                                initializer=self.initializer)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      input_keep_prob=self.config.lstm_dropout_keep_prob,
                                                      output_keep_prob=self.config.lstm_dropout_keep_prob)
            zero_state = lstm_cell.zero_state(
                batch_size=self.config.batch_size, dtype=tf.float32)

            sequence_length = tf.reduce_sum(self.mask, 1)
            _, self.subt_lstm_outputs = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                          inputs=self.subt_embedding,
                                                          sequence_length=sequence_length,
                                                          initial_state=zero_state,
                                                          dtype=tf.float32)


def main(_):
    model = VLLabMemoryModel()
    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    # print('Start extract !!')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        pooled = sess.run(model.pooled_outputs)
        for pool in pooled:
            print(pool)
            print(pool.shape)
        # print(batch_features.shape)
        coord.request_stop()
        coord.join(threads)
        # qe, ae, se = sess.run([model.ques_lstm_outputs,
        #                        model.ans_lstm_outputs,
        #                        model.subt_lstm_outputs])
        #
        # print(qe.h)
        # print(ae.h)
        # print(se.h)
        #
        # print(qe.h.shape, ae.h.shape, se.h.shape)


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

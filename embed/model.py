from functools import partial

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

from utils.model_utils import get_initializer, extract_axis_1


class NGramModel(object):
    def __init__(self, data, oracle):
        self.data = data
        self.initializer = get_initializer(oracle.args['initializer'],
                                           s=oracle.args['initial_scale'])
        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size - 1, 300],
                                           tf.float32, self.initializer, trainable=True)
        zero_vec = tf.get_variable("zero_vec", [1, 300], tf.float32, tf.constant_initializer(0),
                                   trainable=False)
        gram_matrix = tf.concat([embedding_matrix, zero_vec], axis=0)

        self.gram_embedding = tf.nn.embedding_lookup(gram_matrix, self.data.word)

        self.output = tf.reduce_sum(self.gram_embedding, axis=1)


class BasicModel(object):
    def __init__(self, data, initializer):
        self.data = data
        self.init_mean, self.init_stddev = tf.placeholder(tf.float32, shape=[], name='init_mean'), \
                                           tf.placeholder(tf.float32, shape=[], name='init_stddev')
        self.initializer = get_initializer(initializer, self.init_mean, self.init_stddev)


class MatricesModel(BasicModel):
    def __init__(self, data, initializer='xavier', embedding_size=300):
        super(MatricesModel, self).__init__(data, initializer)

        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size, embedding_size, embedding_size],
                                           tf.float32, self.initializer, trainable=True)
        # bias_matrix = tf.get_variable("bias_matrix", [self.data.vocab_size, embedding_size],
        #                               tf.float32, initializer, trainable=True)
        self.char_embedding = tf.transpose(tf.nn.embedding_lookup(embedding_matrix, self.data.word), [1, 0, 2, 3])

        mat_init = tf.get_variable('mat_init', [1, 1, embedding_size], tf.float32, self.initializer)

        self.mat_init = tf.tile(mat_init, [self.data.batch_size, 1, 1])
        print(self.mat_init.shape)

        self.chain_mul = tf.transpose(tf.scan(lambda a, x: tf.matmul(a, x) + a, self.char_embedding,
                                              initializer=self.mat_init),
                                      [1, 0, 2, 3])
        print(self.chain_mul.shape)

        self.output = extract_axis_1(tf.squeeze(self.chain_mul, 2), self.data.len - 1)
        print(self.output.shape)


class MyConvModel(BasicModel):
    def __init__(self, data, char_dim=64, conv_channel=512, initializer='xavier',
                 embedding_size=300):
        super(MyConvModel, self).__init__(data, initializer)

        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size, char_dim],
                                           tf.float32, self.initializer, trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)

        conv_output = []
        for i in range(6):
            conv_output.append(layers.conv2d(self.char_embedding, conv_channel,
                                             [i + 1], padding='SAME', activation_fn=None))
            print(conv_output[i].shape)

        for i in range(6):
            conv_output[i] = layers.maxout(conv_output[i], 1, axis=1)
            print(conv_output[i].shape)
        self.conv_concat = tf.squeeze(tf.concat(conv_output, 2), 1)
        self.fc1 = layers.fully_connected(self.conv_concat, embedding_size, activation_fn=None)
        self.output = layers.fully_connected(self.fc1, embedding_size, activation_fn=None)
        print(self.output.shape)


class MyModel(BasicModel):
    def __init__(self, data, char_dim=64, hidden_dim=256, num_layers=2, initializer='xavier',
                 rnn_cell='GRU', bias_init=0, rnn_class='single'):
        super(MyModel, self).__init__(data, initializer)

        embedding_matrix = tf.get_variable("embedding_matrix", [self.data.vocab_size, char_dim],
                                           tf.float32, self.initializer, trainable=True)
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
        if rnn_cell == 'GRU':
            cell_fn = partial(tf.nn.rnn_cell.GRUCell, num_units=total_units,
                              kernel_initializer=self.initializer,
                              bias_initializer=tf.constant_initializer(bias_init),
                              activation=tf.nn.leaky_relu)
        elif rnn_cell == 'LSTM':
            cell_fn = partial(tf.nn.rnn_cell.LSTMCell, num_units=total_units,
                              initializer=self.initializer,
                              forget_bias=bias_init,
                              activation=tf.nn.leaky_relu)
        elif rnn_cell == 'BasicRNN':
            cell_fn = partial(tf.nn.rnn_cell.BasicRNNCell, num_units=total_units,
                              activation=tf.nn.leaky_relu)
        else:
            cell_fn = None

        if rnn_class == 'multi':
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


# identity / truncated / random / orthogonal/ glorot
class EmbeddingModel(BasicModel):
    def __init__(self, data, char_dim=100, hidden_dim=256, initializer='xavier',
                 rnn_cell='GRU', bias_init=0, rnn_class='single', embedding_size=300):
        super(EmbeddingModel, self).__init__(data, initializer)

        embedding_matrix = tf.get_variable(
            name="embedding_matrix", initializer=self.initializer,
            shape=[self.data.vocab_size, char_dim], trainable=True)
        self.char_embedding = tf.nn.embedding_lookup(embedding_matrix, self.data.word)

        if rnn_cell == 'GRU':
            cell_fn = partial(rnn.GRUCell, num_units=hidden_dim,
                              kernel_initializer=self.initializer,
                              bias_initializer=tf.constant_initializer(bias_init))
        elif rnn_cell == 'LSTM':
            cell_fn = partial(rnn.CoupledInputForgetGateLSTMCell,
                              num_units=hidden_dim, initializer=self.initializer)
        elif rnn_cell == 'BasicRNN':
            cell_fn = partial(tf.nn.rnn_cell.BasicRNNCell, num_units=hidden_dim)
        else:
            cell_fn = None

        if rnn_class == 'multi':
            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(4)])
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(4)])
        else:
            lstm_cell_fw = cell_fn()
            lstm_cell_bw = cell_fn()

        init_fw_c_state = tf.get_variable("init_fw_c_state", [1, hidden_dim], tf.float32, self.initializer)
        init_fw_h_state = tf.get_variable("init_fw_h_state", [1, hidden_dim], tf.float32, self.initializer)
        fw_state = rnn.LSTMStateTuple(tf.tile(init_fw_c_state, [self.data.batch_size, 1]),
                                      tf.tile(init_fw_h_state, [self.data.batch_size, 1]))

        init_bw_c_state = tf.get_variable("init_bw_c_state", [1, hidden_dim], tf.float32, self.initializer)
        init_bw_h_state = tf.get_variable("init_bw_h_state", [1, hidden_dim], tf.float32, self.initializer)
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
        if rnn == 'multi':
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
                                   biases_initializer=tf.constant_initializer(bias_init)), [1, 0]), 2)

        output_list = []
        for i in range(16):
            with tf.variable_scope('Affine_Transform_%d' % i):
                if rnn == 'multi':
                    self.fc = layers.fully_connected(self.fc, 1024)
                fc2 = layers.fully_connected(self.fc, embedding_size, activation_fn=tf.nn.tanh,
                                             biases_initializer=tf.constant_initializer(bias_init))
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

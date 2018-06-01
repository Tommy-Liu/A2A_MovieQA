import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from raw_input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 256, 'feat_dim': 512, 'dropout_rate': 0.1}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def l2_norm(x, axis=1):
    return tf.nn.l2_normalize(x, axis=axis)


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def dense(x, units=hp['emb_dim'], use_bias=True, activation=tf.nn.relu, reuse=False):
    return tf.layers.dense(x, units, activation=activation, use_bias=use_bias, reuse=reuse)


class Model(object):
    def __init__(self, data, beta=0.0, training=False):
        self.data = data
        reg = layers.l2_regularizer(beta)
        initializer = tf.glorot_normal_initializer()

        def group_conv(inp, filters=hp['emb_dim'], split=32):
            conv_list = []
            for i in range(split):
                conv = tf.layers.conv1d(inp, filters / split, 3, kernel_initializer=initializer,
                                        activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
                conv = l2_norm(conv, 2)
                conv = tf.layers.conv1d(conv, filters / split, 3, kernel_initializer=initializer,
                                        activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
                conv = l2_norm(conv, 2)
                conv_list.append(conv)
            conv_output = tf.concat(conv_list, 2)
            # conv_output = l2_norm(conv_output, 2)
            return conv_output

        with tf.variable_scope('Embedding_Linear'):
            self.ques = self.data.ques
            self.ans = self.data.ans
            self.subt = self.data.subt
            self.ques = l2_norm(self.ques)
            self.ans = l2_norm(self.ans)
            self.subt = l2_norm(self.subt)
            # with tf.variable_scope('Question'):
            #     # (1, E_t)
            #     self.ques = tf.layers.dense(self.ques, hp['emb_dim'] / 8, tf.nn.relu,
            #                                 kernel_initializer=initializer, kernel_regularizer=reg)
            #     self.ques = dropout(self.ques, training)
            #     self.ques = l2_norm(self.ques)
            with tf.variable_scope('Answers_Subtitles'):
                # (5, E_t)
                self.ans = tf.layers.dense(self.ans, hp['emb_dim'],  # tf.nn.relu,
                                           kernel_initializer=initializer, kernel_regularizer=reg)
                self.ans = dropout(self.ans, training)
                self.ans = l2_norm(self.ans)
                # (N, E_t)
                self.subt = tf.layers.dense(self.subt, hp['emb_dim'],  # tf.nn.relu,
                                            kernel_initializer=initializer, reuse=True)
                self.subt = dropout(self.subt, training)
                self.subt = l2_norm(self.subt)

            # (1, N, E_t)
            s_exp = tf.expand_dims(self.subt, 0)
            # (1, 1, E_t)
            q_exp = tf.expand_dims(self.ques, 0)
            # (1, 5, E_t)
            a_exp = tf.expand_dims(self.ans, 0)

        s_shape = tf.shape(self.subt)
        with tf.variable_scope('Abstract'):
            # (1, N, E_t)
            self.conv1 = group_conv(s_exp, hp['emb_dim'])
            # (1, N, E_t / 2)
            # self.conv2 = tf.layers.conv1d(self.conv1, hp['emb_dim'] / 8, 3, dilation_rate=2,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv2 = l2_norm(self.conv2, 2)
            # (1, N, E_t / 4)
            # self.conv3 = tf.layers.conv1d(self.conv2, hp['emb_dim'], 3, dilation_rate=3,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv3 = l2_norm(self.conv3, 2)
            # (1, N, E_t / 8)
            # self.conv4 = tf.layers.conv1d(self.conv3, hp['emb_dim'] / 8, 3, dilation_rate=4,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv4 = l2_norm(self.conv4, 2)
            # (1, N, E_t / 16)
            # self.conv5 = tf.layers.conv1d(self.conv4, hp['emb_dim'] / 16, 3, dilation_rate=5,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv5 = l2_norm(self.conv5, 2)

            # (N, E_t, E_t / 8)
            self.confuse = tf.matmul(tf.transpose(s_exp, [1, 2, 0]), tf.transpose(self.conv1, [1, 0, 2]))

            # (E_t / 8, N, E_t)
            self.confuse = tf.transpose(self.confuse, [2, 0, 1])

            # (E_t / 8, 5, N_t)
            self.response = tf.matmul(tf.tile(a_exp, [tf.shape(self.confuse)[0], 1, 1]),
                                      self.confuse, transpose_b=True)
            # (E_t / 8, 5, 8)
            self.top_k_response, _ = tf.nn.top_k(self.response, 2)
            # (5, E_t / 8)
            self.top_k_response = tf.transpose(tf.reduce_sum(self.top_k_response, 2))
            # (5, 2)
            self.top_k_output, _ = tf.nn.top_k(self.top_k_response, 2)
            # (1, 5)
            self.output = tf.transpose(tf.reduce_sum(self.top_k_output, 1, keepdims=True))


def main():
    data = Input(split='train', mode='subt')
    model = Model(data)

    for v in tf.global_variables():
        print(v)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run([model.data.initializer, tf.global_variables_initializer()],
                 feed_dict=data.feed_dict)

        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        # a, b, c, d = sess.run(model.tri_word_encodes)
        # print(a, b, c, d)
        # print(a.shape, b.shape, c.shape, d.shape)
        a, b = sess.run([model.subt, model.output])
        print(a, b)
        print(a.shape, b.shape)


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, nn

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

        def transpose_conv1d(inp, filters, kernel_size, strides=2, padding='SAME', activation=None):
            batch, width, channel = inp.get_shape().as_list()
            w = tf.get_variable('kernel', [kernel_size, channel, filters], initializer=initializer,
                                regularizer=reg)
            b = tf.get_variable('bias', [filters], initializer=tf.zeros_initializer())
            conv = nn.conv1d_transpose(inp, w, [1, tf.shape(inp)[1] * strides, filters], stride=strides,
                                       padding=padding)
            conv = tf.nn.bias_add(conv, b)
            if activation:
                conv = activation(conv)
            return conv

        with tf.variable_scope('Embedding_Linear'):
            self.ques = self.data.ques
            self.ans = self.data.ans
            self.subt = self.data.subt
            self.ques = l2_norm(self.ques)
            self.ans = l2_norm(self.ans)
            self.subt = l2_norm(self.subt)
            with tf.variable_scope('Question'):
                # (1, E_t)
                self.ques = tf.layers.dense(self.ques, hp['emb_dim'], tf.nn.relu,
                                            kernel_initializer=initializer, kernel_regularizer=reg)
                self.ques = dropout(self.ques, training)
                self.ques = l2_norm(self.ques)
            with tf.variable_scope('Answers_Subtitles'):
                # (5, E_t)
                self.ans = tf.layers.dense(self.ans, hp['emb_dim'],  # tf.nn.relu,
                                           kernel_initializer=initializer, kernel_regularizer=reg)
                self.ans = l2_norm(self.ans)
                self.ans = dropout(self.ans, training)
                # (N, E_t)
                self.subt = tf.layers.dense(self.subt, hp['emb_dim'],  # tf.nn.relu,
                                            kernel_initializer=initializer, reuse=True)
                self.subt = l2_norm(self.subt)
                self.subt = dropout(self.subt, training)
            # (1, N, E_t)
            s_exp = tf.expand_dims(self.subt, 0)
            # (1, 1, E_t)
            q_exp = tf.expand_dims(self.ques, 0)
            # (1, 5, E_t)
            a_exp = tf.expand_dims(self.ans, 0)

        s_shape = tf.shape(self.subt)
        with tf.variable_scope('Abstract'):
            # (1, N, E_t)
            self.conv1 = transpose_conv1d(s_exp, hp['emb_dim'], 2, activation=tf.nn.relu,
                                          strides=2)
            self.conv1 = l2_norm(self.conv1, 2)

            self.conv1 = tf.layers.conv1d(self.conv1, hp['emb_dim'], 3, activation=tf.nn.relu, strides=2,
                                          kernel_initializer=initializer, padding='same', kernel_regularizer=reg)
            self.conv1 = l2_norm(self.conv1, 2)

            # (1, N, 1)
            self.attn = tf.reduce_sum(self.conv1 * q_exp, 2, keepdims=True)
            self.attn = tf.squeeze(self.attn, 0)

            # (N, E_t)
            self.confuse = self.subt * self.attn

            # (5, N_t)
            self.response = tf.matmul(self.ans, self.confuse, transpose_b=True)
            # (5, 4)
            self.top_k_response, _ = tf.nn.top_k(self.response, 4)
            # (1, 5)
            self.output = tf.transpose(tf.reduce_sum(self.top_k_response, 1, keepdims=True))


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
        a, b = sess.run([model.subt, model.conv1])
        print(a, b)
        print(a.shape, b[np.sum(b, 2) != 0.0].shape)


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 512, 'feat_dim': 512,
      'learning_rate': 10 ** (-3), 'decay_rate': 1, 'decay_type': 'inv_sqrt', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def make_mask(x, length):
    return tf.tile(tf.expand_dims(tf.sequence_mask(x, maxlen=length),
                                  axis=-1), [1, 1, hp['emb_dim']])


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def safe_mean(x, length):
    length = tf.reshape(tf.to_float(tf.maximum(tf.constant(1, dtype=tf.int64), length)), [-1, 1, 1])
    return tf.reduce_sum(x, axis=1, keepdims=True) / length


def conv_encode(x, reuse=True):
    with tf.variable_scope('conv_encode', reuse=reuse):
        x1 = tf.layers.conv1d(x, filters=hp['emb_dim'], kernel_size=3, padding='same', activation=tf.nn.relu)
        x2 = tf.layers.conv1d(x1, filters=hp['emb_dim'], kernel_size=3, padding='same', activation=tf.nn.relu) + x1
        x3 = tf.layers.conv1d(x2, filters=hp['emb_dim'], kernel_size=3, padding='same', activation=tf.nn.relu) + x2
        x4 = tf.layers.conv1d(x3, filters=hp['emb_dim'], kernel_size=3, padding='same', activation=tf.nn.relu) + x3

        x1 = tf.reduce_max(x1, axis=1, keepdims=True)
        x2 = tf.reduce_max(x2, axis=1, keepdims=True)
        x3 = tf.reduce_max(x3, axis=1, keepdims=True)
        x4 = tf.reduce_max(x4, axis=1, keepdims=True)

        x_c = tf.concat([x1, x2, x3, x4], axis=-1)
    return x_c


class Model(object):
    def __init__(self, data, training=False):
        self.data = data
        self.initializer = tf.orthogonal_initializer()
        q_mask = make_mask(self.data.ql, 25)  # (1, L_q, E)
        s_mask = make_mask(self.data.sl, 29)  # (N, L_s, E)
        a_mask = make_mask(self.data.al, 34)  # (5, L_a, E)

        ques_shape = tf.shape(q_mask)
        subt_shape = tf.shape(s_mask)
        ans_shape = tf.shape(a_mask)

        with tf.variable_scope('Embedding'):
            self.embedding = tf.get_variable('embedding_matrix',
                                             initializer=np.load(_mp.embedding_file), trainable=False)

            self.ques = tf.nn.embedding_lookup(self.embedding, self.data.ques)  # (1, L_q, E)
            self.ans = tf.nn.embedding_lookup(self.embedding, self.data.ans)  # (5, L_a, E)
            self.subt = tf.nn.embedding_lookup(self.embedding, self.data.subt)  # (N, L_s, E)

            # self.ques = dropout(self.ques, training=training)  # (1, L_q, E)
            # self.ans = dropout(self.ans, training=training)  # (5, L_a, E)
            # self.subt = dropout(self.subt, training=training)  # (N, L_s, E)

        # with tf.variable_scope('Embedding_Linear', regularizer=reg):
        #     # (1, L_q, E_t)
        #     self.ques_embedding = unit_norm(tf.layers.dense(self.ques, hp['emb_dim'], activation=None, use_bias=False))
        #     # (5, L_a, E_t)
        #     self.ans_embedding = unit_norm(
        #         tf.layers.dense(self.ans, hp['emb_dim'], activation=None, use_bias=False, reuse=True))
        #     # (N, L_s, E_t)
        #     self.subt_embedding = unit_norm(
        #         tf.layers.dense(self.subt, hp['emb_dim'], activation=None, use_bias=False, reuse=True))

        with tf.variable_scope('Language_Encode'):
            # (1, 1, 4 * E_t)
            self.ques_enc = unit_norm(conv_encode(self.ques, reuse=False))
            # (N, 1, 4 * E_t)
            self.subt_enc = unit_norm(conv_encode(self.subt))
            # (5, 1, 4 * E_t)
            self.ans_enc = unit_norm(conv_encode(self.ans))

        # with tf.variable_scope('Language_Attention'):
        #     shape = tf.shape(self.subt_embedding)
        #     q = tf.tile(self.ques_enc, [shape[0], shape[1], 1])  # (N, L_s, E_t)
        #     q = tf.where(s_mask, q, tf.zeros_like(self.subt_embedding))  # (N, L_s, E_t)
        #     self.sq_concat = tf.concat([self.subt_embedding, q], axis=-1)  # (N, L_s, 2 * E_t)
        #     self.lang_attn = tf.layers.conv1d(self.sq_concat, filters=hp['feat_dim'], kernel_size=3,
        #                                       padding='same', activation=tf.nn.relu)  # (N, L_s, E_t)
        #     self.lang_attn = tf.layers.conv1d(self.lang_attn, filters=1, kernel_size=5, padding='same',
        #                                       dilation_rate=2, activation=None)  # (N, L_s, 1)
        #     self.lang_attn = tf.nn.softmax(self.lang_attn, axis=1)  # (N, L_s, 1)
        #     self.subt_attn_enc = safe_mean(self.subt_embedding * (1 + self.lang_attn), self.data.sl)  # (N, 1, E_t)
        #
        #     alpha = tf.layers.dense(self.ques_enc, 1, activation=tf.nn.sigmoid)  # (1, 1, 1)
        #     self.subt_sum = alpha * self.subt_enc + (1 - alpha) * self.subt_attn_enc  # (N, 1, E_t)
        #

        # with tf.variable_scope('Temporal_Attention'):
        #     self.vs_concat = tf.transpose(tf.concat([self.subt_enc,
        #                                              tf.tile(self.ques_enc, [shape[0], 1, 1])], axis=-1),
        #                                   [1, 0, 2])  # (1, N, 2 * E_t)
        #
        #     self.temp_attn = tf.layers.conv1d(self.vs_concat, filters=hp['feat_dim'], kernel_size=5, padding='same',
        #                                       activation=tf.nn.relu)  # (1, N, E_t)
        #
        #     self.temp_attn = tf.layers.conv1d(self.temp_attn, filters=hp['feat_dim'] / 4, kernel_size=7,
        #                                       dilation_rate=2, padding='same', activation=tf.nn.relu)  # (1, N, E_t / 4)
        #
        #     self.focus1 = tf.layers.conv1d(self.temp_attn, filters=1, kernel_size=9, dilation_rate=3,
        #                                    padding='same', activation=None)  # (1, N, 1)
        #
        #     self.focus2 = tf.layers.conv1d(self.temp_attn, filters=1, kernel_size=9, dilation_rate=3,
        #                                    padding='same', activation=None)  # (1, N, 1)
        #
        #     self.subt_temp1 = tf.transpose(self.subt_sum, [1, 0, 2]) * tf.nn.softmax(self.focus1, axis=1)  # (1, N, E_t)
        #
        #     self.subt_temp2 = tf.transpose(self.subt_sum, [1, 0, 2]) * tf.nn.softmax(self.focus2, axis=1)  # (1, N, E_t)
        #
        # with tf.variable_scope('Answer'):
        #     beta = tf.layers.dense(self.ques_enc, 1, activation=tf.nn.sigmoid)  # (1, 1, 1)
        #
        #     self.summarize = tf.reduce_sum(beta * self.subt_temp1 + (1 - beta) * self.subt_temp2, axis=1)  # (1, E_t)
        #

        self.summarize = unit_norm(tf.reduce_mean(self.subt_enc, axis=0), dim=1)  # (1, 4 * E_t)

        # gamma = tf.get_variable('gamma', [1, 1], initializer=tf.zeros_initializer)

        # self.ans_vec = self.summarize * tf.nn.sigmoid(gamma) + \
        #                tf.squeeze(self.ques_enc, axis=0) * (1 - tf.nn.sigmoid(gamma))

        self.ans_vec = unit_norm(self.summarize + tf.squeeze(self.ques_enc, axis=0), dim=1)  # (1, 4 * E_t)

        self.output = tf.matmul(self.ans_vec, tf.squeeze(self.ans_enc), transpose_b=True)  # (1, 5)


def main():
    data = Input(split='train', mode='subt')
    model = Model(data)

    for v in tf.global_variables():
        print(v)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run([model.data.initializer, tf.global_variables_initializer()], )

        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        # a, b, c, d = sess.run(model.tri_word_encodes)
        # print(a, b, c, d)
        # print(a.shape, b.shape, c.shape, d.shape)
        a, b = sess.run([model.ans_vec, model.output])
        print(a, b)
        print(a.shape, b.shape)


if __name__ == '__main__':
    main()

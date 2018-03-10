import numpy as np
import tensorflow as tf

from config import MovieQAPath
from input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512,
      'learning_rate': 10 ** (-4), 'decay_rate': 0.97, 'decay_type': 'exp', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1, 'pos_len': 35}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def make_mask(x, length):
    return tf.tile(tf.expand_dims(tf.sequence_mask(x, maxlen=length),
                                  axis=-1), [1, 1, hp['emb_dim']])


def sliding(x):
    return tf.nn.pool(x, [3], 'AVG', 'SAME', data_format='NWC')


def seq_mean(x, l):
    return tf.reduce_sum(x, axis=1) / tf.to_float(tf.expand_dims(l, axis=-1))


class Model(object):
    def __init__(self, data, training=False):
        self.data = data
        self.initializer = tf.glorot_normal_initializer()
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

            # self.ques = tf.layers.dropout(self.ques, hp['dropout_rate'], training=training)  # (1, L_q, E)
            # self.ans = tf.layers.dropout(self.ans, hp['dropout_rate'], training=training)  # (5, L_a, E)
            # self.subt = tf.layers.dropout(self.subt, hp['dropout_rate'], training=training)  # (N, L_s, E)

        with tf.variable_scope('Embedding_Linear'):
            self.ques_embedding = self.embedding_linear(self.ques, q_mask, 'question')  # (1, L_q, E_t)
            self.ans_embedding = self.embedding_linear(self.ans, a_mask, 'answer')  # (5, L_a, E_t)
            self.subt_embedding = self.embedding_linear(self.subt, s_mask, 'subtitle')  # (N, L_s, E_t)

        with tf.variable_scope('Language_Attention'):
            position_attn = tf.get_variable('position_attention', shape=[hp['pos_len'], hp['emb_dim']],
                                            initializer=self.initializer, trainable=False)
            ques_pos, _ = tf.split(position_attn, [25, hp['pos_len'] - 25])
            ans_pos, _ = tf.split(position_attn, [34, hp['pos_len'] - 34])
            subt_pos, _ = tf.split(position_attn, [29, hp['pos_len'] - 29])

            self.one_word_encodes = [self.subt_embedding]
            ques_enc = seq_mean(self.ques_embedding * ques_pos, self.data.ql)  # (1, E_t)

            for i in range(2):
                with tf.variable_scope('OneWord_Attention_%d' % i):
                    subt_enc = seq_mean(self.one_word_encodes[-1] * subt_pos, self.data.sl)  # (N, E_t)

                    subt_pre_attn = tf.nn.tanh(
                        self.dense_wo_everything(ques_enc) +
                        self.dense_wo_everything(subt_enc) +
                        self.dense_wo_everything(tf.reduce_mean(subt_enc, axis=0, keepdims=True)))  # (N, E_t)

                    subt_pre_attn = tf.expand_dims(
                        tf.einsum('ijk,ik->ij', self.one_word_encodes[-1], subt_pre_attn),
                        axis=-1)  # (N, L_s, 1)

                    subt_attn = tf.nn.softmax(subt_pre_attn, axis=1)  # (N, L_s, 1)

                    self.one_word_encodes.append(self.one_word_encodes[-1] * (1 + subt_attn))  # (N, L_s, E_t)

            self.one_word_mean = tf.concat([tf.expand_dims(t, axis=0)
                                            for t in self.one_word_encodes],
                                           axis=0)  # (3, N, L_s, E_t)

            self.one_word_weight = tf.reshape(
                tf.nn.softmax(
                    tf.layers.dense(ques_enc, 3, kernel_initializer=self.initializer),
                    axis=-1),
                [3, 1, 1, 1])  # (3, 1, 1, 1)

            self.one_word_mean = tf.transpose(
                tf.reduce_sum(
                    tf.reduce_sum(self.one_word_mean * self.one_word_weight, axis=0),
                    axis=1, keepdims=True)
                / tf.to_float(tf.reshape(self.data.sl, [-1, 1, 1])),
                [1, 0, 2])  # (1, N, E_t)

            self.pool_subt = sliding(self.subt_embedding * subt_pos)  # (N, L_s, E_t)
            self.tri_word_encodes = [self.pool_subt]

            for i in range(2):
                with tf.variable_scope('TriWord_Attention_%d' % i):
                    pool_subt_enc = seq_mean(self.tri_word_encodes[-1], self.data.sl)  # (N, E_t)

                    pool_subt_pre_attn = tf.nn.tanh(
                        self.dense_wo_everything(ques_enc) +
                        self.dense_wo_everything(pool_subt_enc) +
                        self.dense_wo_everything(tf.reduce_mean(pool_subt_enc, axis=0, keepdims=True)))  # (N, E_t)

                    pool_subt_pre_attn = tf.expand_dims(
                        tf.einsum('ijk,ik->ij', self.tri_word_encodes[-1], pool_subt_pre_attn),
                        axis=-1)  # (N, L_s, 1)

                    pool_subt_attn = tf.nn.softmax(pool_subt_pre_attn, axis=1)  # (N, L_s, 1)

                    self.tri_word_encodes.append(self.tri_word_encodes[-1] * (1 + pool_subt_attn))  # (N, L_s, E_t)

            self.tri_word_mean = tf.concat([tf.expand_dims(t, axis=0)
                                            for t in self.tri_word_encodes],
                                           axis=0)  # (3, N, L_s, E_t)

            self.tri_word_weight = tf.reshape(
                tf.nn.softmax(
                    tf.layers.dense(ques_enc, 3, kernel_initializer=self.initializer),
                    axis=-1),
                [3, 1, 1, 1])  # (3, 1, 1, 1)

            self.tri_word_mean = tf.transpose(
                tf.reduce_sum(
                    tf.reduce_sum(self.tri_word_mean * self.tri_word_weight, axis=0),
                    axis=1, keepdims=True)
                / tf.to_float(tf.reshape(self.data.sl, [-1, 1, 1])),
                [1, 0, 2])  # (1, N, E_t)

        tile_ques_enc = tf.tile(tf.expand_dims(ques_enc, axis=0), [1, subt_shape[0], 1])  # (1, N, E_t)

        self.one_concat = tf.concat([self.one_word_mean, tile_ques_enc], axis=-1)  # (1, N, 2 * E_t)

        self.tri_concat = tf.concat([self.tri_word_mean, tile_ques_enc], axis=-1)  # (1, N, 2 * E_t)

        with tf.variable_scope('Temporal_Attention'):
            self.temp_one_attn = tf.nn.softmax(
                tf.layers.conv1d(
                    tf.layers.conv1d(self.one_concat, hp['emb_dim'] * 2, 3, padding='same', activation=tf.nn.relu),
                    1, 5, padding='same', activation=None),
                axis=1)  # (1, N, 1)

            self.temp_tri_attn = tf.nn.softmax(
                tf.layers.conv1d(
                    tf.layers.conv1d(self.tri_concat, hp['emb_dim'] * 2, 3, padding='same', activation=tf.nn.relu),
                    1, 5, padding='same', activation=None
                ),
                axis=1)  # (1, N, 1)

            self.temp_one = tf.reduce_sum(self.one_word_mean * self.temp_one_attn, axis=1)  # (1, E_t)

            self.temp_tri = tf.reduce_sum(self.tri_word_mean * self.temp_tri_attn, axis=1)  # (1, E_t)

            self.temp_weight = tf.transpose(tf.nn.softmax(
                tf.layers.dense(ques_enc, 3, kernel_initializer=self.initializer), axis=-1))  # (3, 1)

            self.ans_vec = tf.concat([self.temp_one, self.temp_tri, ques_enc], axis=0) * self.temp_weight

            self.ans_vec = tf.tile(tf.reduce_sum(self.ans_vec, axis=0, keepdims=True), [5, 1])

        ans_enc = seq_mean(self.ans_embedding * ans_pos, self.data.al)
        self.output = tf.reduce_sum(self.ans_vec * ans_enc, axis=1)

    def embedding_linear(self, x, x_mask, scope):
        with tf.variable_scope(scope):
            x = tf.layers.dense(x, hp['emb_dim'] * 4, activation=tf.nn.relu, kernel_initializer=self.initializer)
            x = tf.layers.dense(x, hp['emb_dim'], kernel_initializer=self.initializer)
            zeros = tf.zeros_like(x)
            x = tf.where(x_mask, x, zeros)
        return x

    def dense_wo_everything(self, x):
        return tf.layers.dense(x, hp['emb_dim'], use_bias=False, kernel_initializer=self.initializer)


def main():
    data = Input(split='train')
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

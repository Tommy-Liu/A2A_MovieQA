import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, nn

from config import MovieQAPath
from legacy.input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 512, 'feat_dim': 512,
      'learning_rate': 10 ** (-3), 'decay_rate': 1, 'decay_type': 'inv_sqrt', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1}

reg = layers.l2_regularizer(0.01)


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def make_mask(x, length):
    return tf.tile(tf.expand_dims(tf.sequence_mask(x, maxlen=length),
                                  axis=-1), [1, 1, hp['emb_dim']])


def mask_tensor(x, mask):
    zeros = tf.zeros_like(x)
    x = tf.where(mask, x, zeros)

    return x


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def safe_mean(x, length):
    length = tf.reshape(tf.to_float(tf.maximum(tf.constant(1, dtype=tf.int64), length)), [-1, 1, 1])
    return tf.reduce_sum(x, axis=1, keepdims=True) / length


def dense(x, units=hp['emb_dim'], reuse=True):
    return tf.layers.dense(x, units, activation=tf.nn.relu, reuse=reuse)


def mask_dense(x, mask, reuse=True):
    return mask_tensor(dense(x, reuse=reuse), mask)


def conv_encode(x, mask, scope):
    with tf.variable_scope(scope):
        attn = tf.layers.conv1d(x, filters=1, kernel_size=3, padding='same', activation=tf.nn.relu)
        attn = tf.where(mask, attn, tf.ones_like(attn) * (-2 ** 32 + 1))
        attn = tf.nn.softmax(attn, axis=1)
    return tf.reduce_sum(x * attn, axis=1)


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

        with tf.variable_scope('Embedding_Linear'):
            # (1, L_q, E_t)
            self.ques_embedding = unit_norm(mask_dense(self.ques, q_mask, reuse=False))
            # (5, L_a, E_t)
            self.ans_embedding = unit_norm(mask_dense(self.ans, a_mask))
            # (N, L_s, E_t)
            self.subt_embedding = unit_norm(mask_dense(self.subt, s_mask))

        with tf.variable_scope('Language_Encode'):
            mask = tf.expand_dims(tf.sequence_mask(self.data.ql, 25), axis=-1)
            self.ques_enc = unit_norm(conv_encode(self.ques_embedding, mask, 'ques'), dim=1)
            mask = tf.expand_dims(tf.sequence_mask(self.data.al, 34), axis=-1)
            self.ans_enc = unit_norm(conv_encode(self.ans_embedding, mask, 'ans'), dim=1)
            mask = tf.expand_dims(tf.sequence_mask(self.data.sl, 29), axis=-1)
            self.subt_enc = unit_norm(conv_encode(self.subt_embedding, mask, 'subt'), dim=1)

        with tf.variable_scope('Temporal_Attention'):
            self.temp_attn = tf.concat([self.subt_enc, tf.tile(self.ques_enc, [subt_shape[0], 1])], axis=-1)
            self.temp_attn = dense(self.temp_attn, reuse=False)
            self.temp_attn = unit_norm(tf.expand_dims(self.temp_attn, axis=0))
            self.temp_attn = tf.layers.conv1d(self.temp_attn, 1, 5, padding='same', activation=tf.nn.relu)
            self.temp_attn = tf.squeeze(tf.nn.softmax(self.temp_attn, axis=1), axis=0)
            # self.subt_enc = tf.expand_dims(self.subt_enc, axis=0) * self.temp_attn
            nth = nn.nth_element(tf.transpose(self.temp_attn), tf.cast(subt_shape[0] / 2, tf.int32), True)
            # (N, 1)
            attn_mask = tf.cast(tf.squeeze(tf.greater_equal(self.temp_attn, nth), axis=1), tf.int32)
            _, self.subt_enc = tf.dynamic_partition(self.subt_enc, attn_mask, 2)
            _, self.temp_attn = tf.dynamic_partition(self.temp_attn, attn_mask, 2)
            self.subt_enc = self.subt_enc * self.temp_attn

        self.summarize = unit_norm(tf.reduce_mean(self.subt_enc, axis=0, keepdims=True), dim=1)  # (1, 4 * E_t)

        # gamma = tf.get_variable('gamma', [1, 1], initializer=tf.zeros_initializer)

        # self.ans_vec = self.summarize * tf.nn.sigmoid(gamma) + \
        #                tf.squeeze(self.ques_enc, axis=0) * (1 - tf.nn.sigmoid(gamma))

        self.ans_vec = unit_norm(self.summarize + self.ques_enc, dim=1)  # (1, 4 * E_t)

        self.output = tf.matmul(self.ans_vec, self.ans_enc, transpose_b=True)  # (1, 5)


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

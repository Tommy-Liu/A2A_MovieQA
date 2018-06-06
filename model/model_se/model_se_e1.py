import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from legacy.input_v2 import Input

_mp = MovieQAPath()
hp = {'emb_dim': 192, 'feat_dim': 512, 'dropout_rate': 0.1}

reg = layers.l2_regularizer(0.1)


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def make_mask(x, length):
    return tf.tile(tf.expand_dims(tf.sequence_mask(x, maxlen=length),
                                  axis=-1), [1, 1, hp['emb_dim']])


def mask_tensor(x, mask):
    zeros = tf.zeros_like(x)
    x = tf.where(mask, x, zeros)

    return x


def l2_norm(x, axis=1):
    return tf.nn.l2_normalize(x, axis=axis)


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def safe_mean(x, length, keepdims=True):
    length = tf.reshape(tf.to_float(tf.maximum(tf.constant(1, dtype=tf.int64), length)), [-1, 1, 1])
    return tf.reduce_sum(x, axis=1, keepdims=keepdims) / length


def dense(x, units=hp['emb_dim'], use_bias=True, activation=tf.nn.relu, reuse=False):
    return tf.layers.dense(x, units, activation=activation, use_bias=use_bias, reuse=reuse)


def mask_dense(x, mask, reuse=True):
    return mask_tensor(dense(x, reuse=reuse), mask)


def conv_encode(x, mask, reuse=True):
    attn = tf.layers.conv1d(x, filters=1, kernel_size=3, padding='same', activation=tf.nn.relu, reuse=reuse)
    attn = tf.where(mask, attn, tf.ones_like(attn) * (-2 ** 32 + 1))
    attn = tf.nn.softmax(attn, axis=1)
    return tf.reduce_sum(x * attn, axis=1)


def dilated_conv_encode(x, mask, reuse=True):
    attn = tf.layers.conv1d(x, filters=1, kernel_size=3, dilation_rate=2,
                            padding='same', activation=tf.nn.relu, reuse=reuse)
    attn = tf.where(mask, attn, tf.ones_like(attn) * (-2 ** 32 + 1))
    attn = tf.nn.softmax(attn, axis=1)
    return tf.reduce_sum(x * attn, axis=1)


def mean_reduce(x, length):
    m = safe_mean(x, length)
    v = tf.matmul(x, tf.transpose(m, [0, 2, 1]))
    x = x - m * v
    return x


def variance_encode(x, length):
    m = safe_mean(x, length)
    v = tf.matmul(x, tf.transpose(m, [0, 2, 1]))
    x = x - m * v
    return tf.reduce_sum(x, axis=1)


def watch_movie(story, mem, l):
    mask = tf.sequence_mask(l, tf.shape(story)[0], dtype=tf.int32)
    _, clips = tf.dynamic_partition(story, mask, 2)


class Model(object):
    def __init__(self, data, training=False):
        self.data = data
        initializer = tf.orthogonal_initializer()
        with tf.variable_scope('Embedding_Linear'):
            self.ques = l2_norm(self.data.ques)
            self.ans = l2_norm(self.data.ans)
            self.subt = l2_norm(self.data.subt)
            with tf.variable_scope('Question'):
                # (1, E_t)
                self.ques = l2_norm(dropout(tf.layers.dense(
                    self.ques, hp['emb_dim'], activation=tf.nn.relu,
                    kernel_initializer=initializer, kernel_regularizer=reg), training))
            with tf.variable_scope('Answers_Subtitles'):
                # (5, E_t)
                self.ans = l2_norm(dropout(tf.layers.dense(
                    self.ans, hp['emb_dim'], tf.nn.relu,
                    kernel_initializer=initializer, kernel_regularizer=reg), training))
                # (N, E_t)
                self.subt = l2_norm(dropout(tf.layers.dense(
                    self.subt, hp['emb_dim'], tf.nn.relu,
                    kernel_initializer=initializer, reuse=True), training))

            # (1, N, E_t)
            s_exp = tf.expand_dims(self.subt, 0)
            # (1, 1, E_t)
            q_exp = tf.expand_dims(self.ques, 0)
            # (1, 5, E_t)
            a_exp = tf.expand_dims(self.ans, 0)

        s_shape = tf.shape(self.subt)
        with tf.variable_scope('Abstract'):
            # (1, N, E_t)
            self.conv1 = l2_norm(
                tf.layers.conv1d(s_exp, hp['emb_dim'], 10, kernel_initializer=initializer,
                                 activation=tf.nn.relu, padding='same', kernel_regularizer=reg), 2)
            # (1, N, E_t / 2)
            self.conv2 = l2_norm(
                tf.layers.conv1d(self.conv1, hp['emb_dim'], 10, dilation_rate=2, kernel_initializer=initializer,
                                 activation=tf.nn.relu, padding='same', kernel_regularizer=reg), 2)
            self.conv2 = l2_norm(self.conv2 + s_exp)
            # (1, N, E_t / 4)
            self.conv3 = l2_norm(
                tf.layers.conv1d(self.conv2, hp['emb_dim'], 10, dilation_rate=4, kernel_initializer=initializer,
                                 activation=tf.nn.relu, padding='same', kernel_regularizer=reg), 2)
            # (1, N, E_t / 8)
            self.conv4 = l2_norm(
                tf.layers.conv1d(self.conv3, hp['emb_dim'], 10, dilation_rate=8, kernel_initializer=initializer,
                                 activation=tf.nn.relu, padding='same', kernel_regularizer=reg), 2)
            self.conv4 = l2_norm(self.conv4 + self.conv2)
            # (N, E_t, E_t / 32)
            self.confuse = tf.matmul(tf.transpose(s_exp, [1, 2, 0]), tf.transpose(self.conv4 * q_exp, [1, 0, 2]))

            # (N, 5, E_t / 32)
            self.response = tf.matmul(tf.tile(a_exp, [tf.shape(self.subt)[0], 1, 1]),
                                      self.confuse)
            # (5, E_t / 32, 2)
            self.top_k_response, _ = tf.nn.top_k(tf.transpose(self.response, [1, 2, 0]), 8)
            # (5, E_t / 32)
            self.top_k_response = tf.reduce_sum(self.top_k_response, 2)
            # (5, 4)
            self.top_k_output, _ = tf.nn.top_k(self.top_k_response, 2)
            # (1, 5)
            self.output = tf.transpose(tf.reduce_sum(self.top_k_output, 1, keepdims=True))
            # self.subt_abst = tf.reduce_max(self.subt, axis=0, keepdims=True)

        # (1, E_t)
        # self.summarize = l2_norm(tf.reduce_sum(self.subt_abst, axis=1))
        # self.summarize = l2_norm(self.subt_abst, axis=1)
        # # (1, E_t)
        # # self.ans_vec = l2_norm(self.summarize + self.ques)
        # self.ans_vec = self.summarize
        # # (1, 5)
        # self.output = tf.matmul(self.ans_vec, self.ans, transpose_b=True)


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
                 feed_dict={data.placeholder: data.files})

        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        # a, b, c, d = sess.run(model.tri_word_encodes)
        # print(a, b, c, d)
        # print(a.shape, b.shape, c.shape, d.shape)
        a, b = sess.run([model.compact_subt, model.subt])
        print(a, b[np.sum(b, axis=1) != 0])
        print(a.shape, b[np.sum(b, axis=1) != 0].shape, b.shape)


if __name__ == '__main__':
    main()

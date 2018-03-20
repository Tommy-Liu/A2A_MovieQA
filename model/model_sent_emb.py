import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512,
      'learning_rate': 10 ** (-1), 'decay_rate': 0.5, 'decay_type': 'exp', 'decay_epoch': 2,
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


class Model(object):
    def __init__(self, data, training=False):
        self.data = data
        self.initializer = layers.xavier_initializer()

        with tf.variable_scope('Embedding_Linear'):
            # (1, L_q, E_t)
            self.ques = l2_norm(tf.layers.dense(
                self.data.ques, hp['emb_dim'], activation=tf.nn.relu,
                kernel_initializer=self.initializer, bias_initializer=self.initializer))
            # (5, L_a, E_t)
            self.ans = l2_norm(tf.layers.dense(
                self.data.ans, hp['emb_dim'], activation=tf.nn.relu,
                kernel_initializer=self.initializer, bias_initializer=self.initializer, reuse=True))
            # (N, L_s, E_t)
            self.subt = l2_norm(tf.layers.dense(
                self.data.subt, hp['emb_dim'], activation=tf.nn.relu,
                kernel_initializer=self.initializer, bias_initializer=self.initializer, reuse=True))

            # self.ques = self.data.ques
            # self.ans = self.data.ans
            # self.subt = self.data.subt

        with tf.variable_scope('Temporal_Attention'):
            # (N, E_t)
            self.temp_attn = tf.layers.dense(self.ques, hp['emb_dim'], use_bias=False, activation=None,
                                             kernel_initializer=self.initializer) + \
                             tf.layers.dense(self.subt, hp['emb_dim'], use_bias=False, activation=None,
                                             kernel_initializer=self.initializer)

            self.temp_attn = tf.layers.dense(
                layers.bias_add(self.temp_attn, tf.nn.tanh, initializer=self.initializer),
                units=1, use_bias=False, activation=None, kernel_initializer=self.initializer)

            self.temp_attn = tf.nn.softmax(self.temp_attn, axis=1)

            # amount = tf.squeeze(dense(self.ques_enc, 1, tf.nn.sigmoid, reuse=False))
            # nth = nn.nth_element(tf.transpose(self.temp_attn),
            #                      tf.cast(tf.cast(subt_shape[0], tf.float32) * amount, tf.int32), True)
            # # (N, 1)
            # attn_mask = tf.cast(tf.squeeze(tf.greater_equal(self.temp_attn, nth), axis=1), tf.int32)
            # _, self.subt_enc = tf.dynamic_partition(self.subt_enc, attn_mask, 2)
            # _, self.temp_attn = tf.dynamic_partition(self.temp_attn, attn_mask, 2)
            self.subt_temp = self.subt * self.temp_attn

        self.summarize = l2_norm(tf.reduce_sum(self.subt_temp, axis=0, keepdims=True))  # (1, E_t)

        # gamma = tf.get_variable('gamma', [1, 1], initializer=tf.zeros_initializer)

        # self.ans_vec = self.summarize * tf.nn.sigmoid(gamma) + \
        #                tf.squeeze(self.ques_enc, axis=0) * (1 - tf.nn.sigmoid(gamma))

        self.ans_vec = l2_norm(self.summarize + self.ques)  # (1, E_t)

        self.output = tf.matmul(self.ans_vec, self.ans, transpose_b=True)  # (1, 5)


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

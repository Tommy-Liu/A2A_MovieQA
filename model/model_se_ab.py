import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512, 'sec_size': 30,
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
                self.data.ques, hp['emb_dim'], kernel_initializer=self.initializer, bias_initializer=self.initializer))
            # (5, L_a, E_t)
            self.ans = l2_norm(tf.layers.dense(
                self.data.ans, hp['emb_dim'], kernel_initializer=self.initializer,
                bias_initializer=self.initializer, reuse=True))
            # (N, L_s, E_t)
            self.subt = l2_norm(tf.layers.dense(
                self.data.subt, hp['emb_dim'],
                kernel_initializer=self.initializer, bias_initializer=self.initializer))

            # self.ques = self.data.ques
            # self.ans = self.data.ans
            # self.subt = self.data.subt

        t_shape = tf.shape(self.subt)
        split_num = tf.to_int32(tf.ceil(t_shape[0] / hp['sec_size']))
        pad_num = split_num * hp['sec_size'] - t_shape[0]
        paddings = tf.convert_to_tensor([[0, pad_num], [0, 0]])
        with tf.variable_scope('Temporal_Attention'):
            # (1, N+p, E_t)
            self.pad_subt = tf.expand_dims(tf.pad(self.subt, paddings), axis=0)
            # (S, (N+p) / S, E_t)
            self.pad_subt = tf.reshape(self.pad_subt, [-1, hp['sec_size'], hp['emb_dim']])
            # (S, 1, E_t)
            self.sec_repr = tf.reduce_mean(self.pad_subt, axis=1, keepdims=True)
            # (S, 1, 1)
            self.sec_score = tf.matmul(tf.layers.dense(self.sec_repr, hp['emb_dim'],
                                                       use_bias=False, kernel_initializer=self.initializer),
                                       tf.tile(tf.reshape(self.ques, [1, -1, 1]), [split_num, 1, 1]))
            # (S, 1, 1)
            self.sec_attn = tf.nn.softmax(self.sec_score, axis=0)

            # (S, (N+p) / S, 1)
            self.local_score = tf.matmul(tf.layers.dense(self.pad_subt, hp['emb_dim'],
                                                         use_bias=False, kernel_initializer=self.initializer),
                                         tf.tile(tf.reshape(self.ques, [1, -1, 1]), [split_num, 1, 1]))
            # (S, (N+p) / S, 1)
            self.local_attn = tf.nn.softmax(self.local_score, axis=1)
            # (1, N+p, E_t)
            self.temp_output = tf.reshape(self.pad_subt * self.sec_attn * self.local_attn, [1, -1, hp['emb_dim']])
        # (1, E_t)
        self.summarize = l2_norm(tf.reduce_sum(self.temp_output, axis=1))
        # (1, E_t)
        self.ans_vec = l2_norm(self.summarize + self.ques)
        # (1, 5)
        self.output = tf.matmul(self.ans_vec, self.ans, transpose_b=True)


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

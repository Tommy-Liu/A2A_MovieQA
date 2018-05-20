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


class Model(object):
    def __init__(self, data, beta=0.0, training=False):
        self.data = data
        reg = layers.l2_regularizer(beta)
        initializer = tf.glorot_normal_initializer(seed=0)

        def dense(x, units=hp['emb_dim'], activation=None, reuse=False, drop=True, use_bias=True, norm=True):
            x = tf.layers.dense(x, units, activation, use_bias=use_bias,
                                kernel_initializer=initializer, kernel_regularizer=reg,
                                reuse=reuse)
            if norm:
                x = l2_norm(x)
            if drop:
                x = dropout(x, training)
            return x

        t = 2
        with tf.variable_scope('Embedding_Linear'):
            self.ques = self.data.ques
            self.ans = self.data.ans
            self.subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
            self.ques = l2_norm(self.ques)
            self.ans = l2_norm(self.ans)
            self.subt = l2_norm(self.subt)
            with tf.variable_scope('Answers_Subtitles'):
                # (5, t * E_t)
                self.ans = dense(self.ans, hp['emb_dim'] * t, tf.nn.tanh)
                # (N, t * E_t)
                self.subt = dense(self.subt, hp['emb_dim'] * t, tf.nn.tanh)
                # (1, t * E_t)
                self.ques = dense(self.ques, hp['emb_dim'] * t, tf.nn.tanh)
                # (t, 5, E_t)
                self.ans = tf.stack(tf.split(self.ans, t, 1))
                # (t, N, E_t)
                self.subt = tf.stack(tf.split(self.subt, t, 1))
                # (t, 1, E_t)
                self.ques = tf.stack(tf.split(self.ques, t, 1))
                # (t, N, 1, E_t)
                self.s_exp = tf.expand_dims(self.subt, 2)

        # with tf.variable_scope('RNN'):
        #     cell = tf.nn.rnn_cell.LSTMCell(hp['emb_dim'], initializer=initializer)
        #     self.rnn_output, _ = tf.nn.dynamic_rnn(cell, s_exp, dtype=tf.float32)
        #     self.rnn_output = tf.squeeze(self.rnn_output, 0)
        #     self.rnn_output = dense(self.rnn_output, 1, tf.nn.tanh, drop=False, norm=False)
        #     self.rnn_attn = tf.nn.softmax(self.rnn_output, 0)

        # with tf.variable_scope('CNN'):
        #     # (1, N, E_t)
        #     self.cnn_subt = tf.layers.conv1d(s_exp, hp['emb_dim'], 2, padding='same', activation=tf.nn.relu,
        #                                      kernel_initializer=initializer, kernel_regularizer=reg)
        #     self.cnn_subt = tf.layers.batch_normalization(self.cnn_subt)
        #     self.cnn_subt = tf.squeeze(self.cnn_subt, 0)
        #     self.subt = self.subt + self.cnn_subt
        #     self.subt = l2_norm(self.subt, 1)
        #     # (N, 1)
        #     s_exp = tf.expand_dims(self.subt, 0)

        with tf.variable_scope('Response'):
            # (t, N, 1)
            self.sq = tf.matmul(self.subt, self.ques, transpose_b=True)
            self.sq = tf.nn.softmax(self.sq, axis=1)
            # (t, N, 5)
            self.sa = tf.matmul(self.subt, self.ans, transpose_b=True)
            self.sa = tf.nn.softmax(self.sa, axis=1)
            # (t, N, 5)
            self.attn = self.sq * self.sa  # * self.rnn_attn
            # (t, N, 5, 1)
            self.attn = tf.expand_dims(self.attn, -1)
            # (t, N, 5, E_t)
            self.abs = self.s_exp * self.attn
            # (t, 5, E_t)
            self.abs = tf.reduce_sum(self.abs, axis=1)
            self.abs = l2_norm(self.abs, 2)
            # (5, E_t)
            self.abs = l2_norm(tf.reduce_sum(self.abs, 0))
            self.ans = l2_norm(tf.reduce_sum(self.ans, 0))

            # (5, 1)
            self.output = tf.reduce_sum(self.abs * self.ans, axis=1, keepdims=True)
            # (5, 1)
            # self.output = tf.reduce_sum(self.output, axis=0)
            # (1, 5)
            self.output = tf.transpose(self.output)


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

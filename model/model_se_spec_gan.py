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
        initializer = tf.glorot_normal_initializer(seed=0)

        def discriminator(x, reuse=False):
            logit = tf.layers.dense(x, 1, kernel_initializer=initializer, kernel_regularizer=reg, reuse=reuse)
            prob = tf.nn.sigmoid(logit)
            return prob, logit

        with tf.variable_scope('QA'):
            with tf.variable_scope('Embedding_Linear'):
                self.ques = self.data.ques
                self.ans = self.data.ans
                self.subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
                self.ques = l2_norm(self.ques)
                self.ans = l2_norm(self.ans)
                self.subt = l2_norm(self.subt)
                with tf.variable_scope('Answers_Subtitles'):
                    # (5, E_t)
                    self.ans = tf.layers.dense(self.ans, hp['emb_dim'],  # tf.nn.relu,
                                               kernel_initializer=initializer, kernel_regularizer=reg)
                    self.ans = l2_norm(self.ans)
                    self.ans = dropout(self.ans, training)
                    # (N, E_t)
                    self.subt = tf.layers.dense(self.subt, hp['emb_dim'],  # tf.nn.relu,
                                                kernel_initializer=initializer, kernel_regularizer=reg)
                    self.subt = l2_norm(self.subt)
                    self.subt = dropout(self.subt, training)
                    # (1, E_t)
                    self.ques = tf.layers.dense(self.ques, hp['emb_dim'],  # tf.nn.relu,
                                                kernel_initializer=initializer, kernel_regularizer=reg)
                    self.ques = l2_norm(self.ques)
                    self.ques = dropout(self.ques, training)
                # (1, N, E_t)
                s_exp = tf.expand_dims(self.subt, 0)
                # (1, 1, E_t)
                q_exp = tf.expand_dims(self.ques, 0)
                # (1, 5, E_t)
                a_exp = tf.expand_dims(self.ans, 0)

            # (N, 1)
            self.sq = tf.matmul(self.subt, self.ques, transpose_b=True)
            self.sq = tf.nn.softmax(self.sq, axis=0)
            # (N, 5)
            self.sa = tf.matmul(self.subt, self.ans, transpose_b=True)
            self.sa = tf.nn.softmax(self.sa, axis=0)
            # (N, 5)
            self.attn = self.sq * self.sa
            # (1, N, 5)
            self.attn = tf.expand_dims(self.attn, 0)
            # (5, N, 1)
            self.attn = tf.transpose(self.attn, [2, 1, 0])
            # (5, N, E_t)
            self.abs = s_exp * self.attn
            # (5, E_t)
            self.abs = tf.reduce_sum(self.abs, axis=1)
            self.abs = l2_norm(self.abs, 1)
            # (5, 1)
            self.output = tf.reduce_sum(self.abs * self.ans, axis=1, keepdims=True)
            # (1, 5)
            self.output = tf.transpose(self.output)
        with tf.variable_scope('Discriminator'):
            self.real, self.real_logit = discriminator(self.ans)
            self.fake, self.fake_logit = discriminator(self.abs, reuse=True)


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

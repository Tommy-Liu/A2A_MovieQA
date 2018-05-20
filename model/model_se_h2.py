import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from raw_input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 256, 'feat_dim': 512, 'dropout_rate': 0.1}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def l2_norm(x, axis=None):
    if axis is None:
        axis = 1
    return tf.nn.l2_normalize(x, axis=axis)


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def dense(x, units=hp['emb_dim'], use_bias=True, activation=tf.nn.relu, reuse=False):
    return tf.layers.dense(x, units, activation=activation, use_bias=use_bias, reuse=reuse)


class Model(object):
    def __init__(self, data, beta=0.0, training=False):
        self.data = data
        reg = layers.l2_regularizer(beta)
        initializer = tf.orthogonal_initializer()

        def dense_kernel(inp, width, in_c, out_c, factor=1, name='', reuse=False):
            with tf.variable_scope('Dense_Kernel_' + name, reuse=reuse):
                k1 = tf.layers.dense(inp, in_c * width * factor, tf.nn.relu,
                                     kernel_initializer=initializer, kernel_regularizer=reg)
                # k1 = dropout(k1, training)
                k1 = tf.reshape(k1, [width, in_c, factor])
                k2 = tf.layers.dense(inp, out_c * width * factor, tf.nn.relu,
                                     kernel_initializer=initializer, kernel_regularizer=reg)
                # k2 = dropout(k2, training)
                k2 = tf.reshape(k2, [width, factor, out_c])
                k = tf.matmul(k1, k2)
                k = l2_norm(k, [0, 1])
                b = tf.get_variable('bias', [out_c], initializer=tf.zeros_initializer())
                return k, b

        # def dense_bias(inp, out_c):
        #     k = tf.layers.dense(inp, out_c,  # tf.nn.relu,
        #                         kernel_initializer=initializer, kernel_regularizer=reg)
        #     # k = dropout(k, training)
        #     k = tf.reshape(k, [out_c])
        #     return k

        with tf.variable_scope('Embedding_Linear'):
            self.ques = self.data.ques
            self.ans = self.data.ans
            self.subt = self.data.subt
            self.ques = l2_norm(self.ques)
            self.ans = l2_norm(self.ans)
            self.subt = l2_norm(self.subt)
            with tf.variable_scope('Question'):
                self.ques = tf.layers.dense(self.ques, hp['emb_dim'], tf.nn.relu,
                                            kernel_initializer=initializer, kernel_regularizer=reg)
                self.ques = dropout(self.ques, training)
                self.ques = l2_norm(self.ques)
                # (3, E_t)
                q_k_1, q_b_1 = dense_kernel(self.ques, 3, hp['emb_dim'], 1, 1, 'q1')
                # q_b_4 = dense_bias(self.ques, hp['emb_dim'] // 8)
            with tf.variable_scope('Answers_Subtitles'):
                # (5, E_t)
                self.ans = tf.layers.dense(self.ans, hp['emb_dim'], tf.nn.relu,
                                           kernel_initializer=initializer, kernel_regularizer=reg)
                self.ans = dropout(self.ans, training)
                self.ans = l2_norm(self.ans)
                a_k_1, a_b_1 = [], []
                for i in range(5):
                    k, b = dense_kernel(tf.expand_dims(self.ans[i], 0), 3, hp['emb_dim'], 1, 1, 'a1', i != 0)
                    a_k_1.append(k)
                    a_b_1.append(b)

                # (N, E_t)
                self.subt = tf.layers.dense(self.subt, hp['emb_dim'], tf.nn.relu,
                                            kernel_initializer=initializer, reuse=True)
                self.subt = dropout(self.subt, training)
                self.subt = l2_norm(self.subt)

            # (1, N, E_t)
            s_exp = tf.expand_dims(self.subt, 0)
            # (1, 1, E_t)
            q_exp = tf.expand_dims(self.ques, 0)
            # (1, 5, E_t)
            a_exp = tf.expand_dims(self.ans, 0)

        with tf.variable_scope('Abstract'):
            # (1, N, 1)
            self.q_attn_1 = tf.nn.bias_add(tf.nn.convolution(
                s_exp, q_k_1, strides=[1], padding='SAME'), q_b_1)

            self.a_attn_1 = []
            for i in range(5):
                a_attn = tf.nn.bias_add(tf.nn.convolution(
                    s_exp, a_k_1[i], strides=[1], padding='SAME'), a_b_1[i])
                self.a_attn_1.append(a_attn)
            # (5, N, 1)
            self.a_attn_1 = tf.concat(self.a_attn_1, 0)

            # (5, N, 1)
            self.attn = tf.nn.sigmoid(self.q_attn_1) * tf.nn.sigmoid(self.a_attn_1)
            self.attn_mask = tf.where(self.attn > 0.5, self.attn, tf.zeros_like(self.attn))

            # (5, N, E_t)
            self.abs = s_exp * self.attn_mask
            # (5, E_t)
            self.abs = l2_norm(tf.reduce_sum(self.abs, axis=1), axis=1)

            # (5, 1)
            self.output = tf.reduce_sum(self.abs * self.ans, axis=1, keepdims=True)
            # (1, 5)
            self.output = tf.transpose(self.output)


def main():
    data = Input(split='train', mode='subt')
    model = Model(data)

    attn = tf.squeeze(model.attn[data.gt[0]])
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
        a, b = sess.run([model.output, attn])
        print(a, b)
        print(a.shape, b.shape)


if __name__ == '__main__':
    main()

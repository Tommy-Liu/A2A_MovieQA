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
        with tf.variable_scope('Embedding_Linear'):
            self.ques = self.data.ques
            self.ans = self.data.ans
            self.subt = self.data.subt
            self.ques = l2_norm(self.ques)
            self.ans = l2_norm(self.ans)
            self.subt = l2_norm(self.subt)
            # with tf.variable_scope('Question'):
            #     # (1, E_t)
            #     self.temp_th = tf.layers.dense(self.ques, 1, tf.nn.sigmoid,
            #                                    kernel_initializer=initializer, kernel_regularizer=reg)
            #     self.mode_th = tf.layers.dense(self.ques, 1, tf.nn.sigmoid,
            #                                    kernel_initializer=initializer, kernel_regularizer=reg)
            #     self.ques = dropout(self.ques, training)
            #     self.ques = l2_norm(self.ques)
            with tf.variable_scope('Answers_Subtitles'):
                # (5, E_t)
                self.ans = tf.layers.dense(self.ans, hp['emb_dim'], tf.nn.relu,
                                           kernel_initializer=initializer, kernel_regularizer=reg)
                self.ans = l2_norm(self.ans)
                self.ans = dropout(self.ans, training)
                # (N, E_t)
                self.subt = tf.layers.dense(self.subt, hp['emb_dim'], tf.nn.relu,
                                            kernel_initializer=initializer, reuse=True)
                self.subt = l2_norm(self.subt)
                self.subt = dropout(self.subt, training)
                # (1, E_t)
                self.ques = tf.layers.dense(self.ques, hp['emb_dim'], tf.nn.relu,
                                            kernel_initializer=initializer, reuse=True)
                self.ques = l2_norm(self.ques)
                self.ques = dropout(self.ques, training)
            # (1, N, E_t)
            s_exp = tf.expand_dims(self.subt, 0)
            # (1, 1, E_t)
            q_exp = tf.expand_dims(self.ques, 0)
            # (1, 5, E_t)
            a_exp = tf.expand_dims(self.ans, 0)

        s_shape = tf.shape(self.subt)

        # with tf.variable_scope('Read'):
        #     self.read1 = tf.layers.conv1d(s_exp, hp['emb_dim'], 3, padding='same',
        #                                   activation=tf.nn.relu,
        #                                   kernel_initializer=initializer, kernel_regularizer=reg)
        #     self.read1 = l2_norm(self.read1, 2)
        #     self.read1 = tf.layers.conv1d(self.read1, hp['emb_dim'], 6, strides=2, padding='same',
        #                                   activation=tf.nn.relu,
        #                                   kernel_initializer=initializer, kernel_regularizer=reg)
        #     self.read1 = l2_norm(self.read1, 2)
        #     self.read1 = tf.layers.conv1d(self.read1, hp['emb_dim'], 9, strides=3, padding='same',
        #                                   activation=tf.nn.relu,
        #                                   kernel_initializer=initializer, kernel_regularizer=reg)
        #     self.read1 = l2_norm(self.read1, 2)

        with tf.variable_scope('Abstract'):
            inp = s_exp
            # (1, N, E_a)
            self.conv1 = tf.layers.conv1d(inp, 256, 3,
                                          kernel_initializer=initializer, activation=tf.nn.relu,
                                          padding='same', kernel_regularizer=reg)
            self.conv1 = l2_norm(self.conv1, 2)
            # # (1, N, E_t / 2)
            # self.conv2 = tf.layers.conv1d(self.conv1, hp['emb_dim'] / 2, 3, dilation_rate=2,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv2 = l2_norm(self.conv2, 2)
            # # (1, N, E_t / 4)
            # self.conv3 = tf.layers.conv1d(self.conv2, hp['emb_dim'] / 4, 3, dilation_rate=3,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv3 = l2_norm(self.conv3, 2)
            # # (1, N, E_t / 8)
            # self.conv4 = tf.layers.conv1d(self.conv3, hp['emb_dim'] / 8, 3, dilation_rate=4,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv4 = l2_norm(self.conv4, 2)
            # (1, N, E_t / 16)
            # self.conv5 = tf.layers.conv1d(self.conv4, hp['emb_dim'] / 16, 3, dilation_rate=5,
            #                               kernel_initializer=initializer,
            #                               activation=tf.nn.relu, padding='same', kernel_regularizer=reg)
            # self.conv5 = l2_norm(self.conv5, 2)

            # (1, N, 5)
            self.confuse = tf.matmul(inp, a_exp, transpose_b=True)
            # (N, 5, 1)
            self.confuse = tf.transpose(self.confuse, [1, 2, 0])
            # (N, 1, E_a)
            self.conv1 = tf.transpose(self.conv1, [1, 0, 2])
            # (N, 5, E_a)
            self.response = tf.matmul(self.confuse, self.conv1)
            # (5, E_a, N)
            self.response = tf.transpose(self.response, [1, 2, 0])

            # (5, E_a, 2)
            self.top_k_response, _ = tf.nn.top_k(self.response, 2)
            # (5, E_a)
            self.top_k_response = tf.reduce_sum(self.top_k_response, 2)
            # (5, 2)
            self.top_k_output, _ = tf.nn.top_k(self.top_k_response, 2)
            # self.top_k_output = self.top_k_response
            # (1, 5)
            self.output = tf.transpose(tf.reduce_sum(self.top_k_output, 1, keepdims=True))


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

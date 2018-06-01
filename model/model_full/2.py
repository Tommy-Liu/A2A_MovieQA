import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from raw_input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512, 'dropout_rate': 0.1}

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def l2_norm(x, axis=1):
    return tf.nn.l2_normalize(x, axis=axis)


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def l1_norm(x, axis=None, epsilon=1e-6, name=None):
    with tf.name_scope(name, "l1_normalize", [x]) as name:
        x = tf.convert_to_tensor(x, name="x")
        square_sum = tf.reduce_sum(x, axis, keepdims=True)
        x_inv_norm = tf.reciprocal(tf.maximum(square_sum, epsilon))
        return tf.multiply(x, x_inv_norm, name=name)


def bhattacharyya_norm(x, axis=None, epsilon=1e-6, name=None):
    with tf.name_scope(name, "l1_normalize", [x]) as name:
        x = tf.convert_to_tensor(x, name="x")
        x = tf.sqrt(x)
        square_sum = tf.reduce_sum(x, axis, keepdims=True)
        x_inv_norm = tf.reciprocal(tf.maximum(square_sum, epsilon))
        return tf.multiply(x, x_inv_norm, name=name)


def cond(i, s, q, a, w, v):
    return i < 10


def body(s, q, a, w, v):
    clue = tf.concat([q, v], axis=1)
    pick = tf.nn.relu(tf.matmul(clue, w))
    v = v
    # v = l2_norm(v)
    return s, q, w, v


def scan_fn(a, x):
    # v, w
    x = tf.expand_dims(x, 0)
    clue = tf.concat([x, a[0]], axis=1)
    pick = tf.nn.relu(tf.matmul(clue, a[1]))
    return a[0] + pick * x, a[1]


def iterable(o):
    try:
        iter(o)
        return True
    except TypeError:
        return False


def pad_and_reshape(x, num):
    length = tf.shape(x)[0]
    split_num = tf.to_int32(tf.ceil(length / num))
    x = tf.pad(x, [[0, split_num * num - length], [0, 0]])
    x = tf.reshape(x, [split_num, 3, hp['emb_dim']])
    return x


class Model(object):
    def __init__(self, data, scale=0.0, training=False):
        self.data = data
        reg = layers.l2_regularizer(scale)
        init = tf.glorot_normal_initializer(seed=0)

        def distance_matrix(n):
            # gamma = tf.get_variable('gamma', [], initializer=tf.ones_initializer())
            index = tf.expand_dims(tf.range(n), 1)
            index_t = tf.transpose(index)
            dist_mat = tf.abs(index - index_t) + 1
            dist_mat = tf.to_float(dist_mat)
            return tf.sqrt(dist_mat)

        with tf.variable_scope('Embedding_Linear'):
            self.raw_ques = self.data.ques
            self.raw_ans = self.data.ans
            # self.raw_subt = self.data.subt
            # self.raw_feat = self.data.subt
            self.raw_subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
            self.raw_feat = tf.boolean_mask(self.data.feat, tf.cast(self.data.spec, tf.bool))
            self.raw_ques = l2_norm(self.raw_ques)
            self.raw_ans = l2_norm(self.raw_ans)
            self.raw_subt = l2_norm(self.raw_subt)
            self.raw_feat = l2_norm(self.raw_feat)

            self.raw_ques = dropout(self.raw_ques, training)
            self.raw_ans = dropout(self.raw_ans, training)
            self.raw_subt = dropout(self.raw_subt, training)
            self.raw_feat = dropout(self.raw_feat, training)

            # (5, E_t)
            self.ans = tf.layers.dense(self.raw_ans, hp['emb_dim'],
                                       kernel_initializer=init, kernel_regularizer=reg)
            self.ans = self.ans + self.raw_ans
            self.ans = l2_norm(self.ans)
            self.ans = dropout(self.ans, training)

            # (N, E_t)
            self.subt = tf.layers.dense(self.raw_subt, hp['emb_dim'],  # reuse=True,
                                        kernel_initializer=init, kernel_regularizer=reg)
            self.subt = self.subt + self.raw_subt
            self.subt = l2_norm(self.subt)
            self.subt = dropout(self.subt, training)

            # (1, E_t)
            self.ques = tf.layers.dense(self.raw_ques, hp['emb_dim'],  # reuse=True,
                                        kernel_initializer=init, kernel_regularizer=reg)
            self.ques = self.ques + self.raw_ques
            self.ques = l2_norm(self.ques)
            self.ques = dropout(self.ques, training)

            self.feat = tf.layers.dense(self.raw_feat, hp['emb_dim'],  # reuse=True,
                                        kernel_initializer=init, kernel_regularizer=reg)
            self.feat = l2_norm(self.feat)
            self.feat = dropout(self.feat, training)

            # num_subt = tf.shape(self.subt)[0]

        with tf.variable_scope('Response'):
            self.fq = tf.matmul(self.feat, tf.tile(tf.expand_dims(self.ques, 0), [tf.shape(self.feat)[0], 1, 1]),
                                transpose_b=True)
            self.fq = tf.nn.relu(self.fq)
            self.fq = dropout(self.fq, training)

            self.feat_new = tf.reduce_sum(self.feat * self.fq, axis=1)
            self.feat_new = l2_norm(self.feat_new)
            self.feat_new = dropout(self.feat_new, training)

            self.fa = tf.matmul(self.feat_new, self.ans, transpose_b=True)
            self.fa = tf.nn.relu(self.fa)
            self.fa = dropout(self.fa, training)

            self.sq = tf.matmul(self.subt, self.ques, transpose_b=True)
            self.sq = tf.nn.relu(self.sq)
            self.sq = dropout(self.sq, training)

            self.sa = tf.matmul(self.subt, self.ans, transpose_b=True)
            self.sa = tf.nn.relu(self.sa)
            self.sa = dropout(self.sa, training)
            # (N_s, 5)
            alpha1 = tf.nn.sigmoid(tf.get_variable('alpha1', [], initializer=tf.zeros_initializer()))
            alpha2 = tf.nn.sigmoid(tf.get_variable('alpha2', [], initializer=tf.zeros_initializer()))
            self.attn = self.sq + alpha1 * self.sa + alpha2 * self.fa
            self.attn = tf.expand_dims(self.attn, 0)
            self.attn = tf.transpose(self.attn, [2, 1, 0])

            self.abst = tf.reduce_sum(self.attn * tf.expand_dims(self.subt, 0), axis=1)
            self.abst_feat = tf.reduce_sum(self.attn * tf.expand_dims(self.feat_new, 0), axis=1)
            beta1 = tf.nn.sigmoid(tf.get_variable('beta1', [], initializer=tf.zeros_initializer()))
            beta2 = tf.nn.sigmoid(tf.get_variable('beta2', [], initializer=tf.zeros_initializer()))
            self.abst = beta1 * self.abst + self.ques * (1 - beta1) + beta2 * self.abst_feat
            self.abst = l2_norm(self.abst)
            self.abst = dropout(self.abst, training)

            self.output = tf.reduce_sum(self.ans * self.abst, axis=1, keepdims=True)
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
        a, b = sess.run([model.subt_1_re, model.subt_1])
        print(a, b)
        print(a.shape, b.shape)
        print(np.array_equal(a[:3], np.squeeze(b[0])))


if __name__ == '__main__':
    main()

import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from raw_input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512, 'dropout_rate': 0.1}


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


class Model(object):
    def __init__(self, data, beta=0.0, training=False):
        self.data = data
        reg = layers.l2_regularizer(beta)
        # reg = layers.l1_regularizer(beta)
        initializer = tf.glorot_normal_initializer(seed=0)
        # constraint = tf.keras.constraints.NonNeg()
        group = 32
        skip = True
        norm = True
        drop = True
        with tf.variable_scope('Embedding_Linear'):
            self.raw_ques = self.data.ques
            self.raw_ans = self.data.ans
            # self.raw_subt = self.data.subt
            self.raw_subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
            self.raw_ques = l2_norm(self.raw_ques)
            self.raw_ans = l2_norm(self.raw_ans)
            self.raw_subt = l2_norm(self.raw_subt)

            self.q_pass = tf.layers.dense(self.raw_ques, group, tf.nn.relu, True, initializer,
                                          kernel_regularizer=reg)
            self.q_pass = tf.reshape(self.q_pass, [group, 1, 1])
            self.q_pass = dropout(self.q_pass, training)
            self.a_pass = tf.layers.dense(self.raw_ques, group, tf.nn.relu, True, initializer,
                                          kernel_regularizer=reg)
            self.a_pass = tf.reshape(self.a_pass, [group, 1, 1])
            self.a_pass = dropout(self.a_pass, training)
            self.s_pass = tf.layers.dense(self.raw_ques, group, tf.nn.relu, True, initializer,
                                          kernel_regularizer=reg)
            self.s_pass = tf.reshape(self.s_pass, [group, 1, 1])
            self.s_pass = dropout(self.s_pass, training)
            # for i in range(5):
            #     self.group_pass = self.group_pass + \
            #                       tf.layers.dense(self.raw_ans[i], tf.nn.relu, True, initializer,
            #                                       kernel_regularizer=reg)

            self.ques = tf.layers.dense(self.raw_ques, hp['emb_dim'] * group, tf.nn.tanh, True, initializer,
                                        kernel_regularizer=reg)
            self.ques = tf.split(self.ques, group, axis=1)
            self.ques = tf.stack(self.ques)
            self.ques = dropout(self.ques, training)
            self.ques = tf.reduce_sum(self.ques * self.q_pass, axis=0)

            self.ans = tf.layers.dense(self.raw_ans, hp['emb_dim'] * group, tf.nn.tanh, True, initializer,
                                       kernel_regularizer=reg)
            self.ans = tf.split(self.ans, group, axis=1)
            self.ans = tf.stack(self.ans)
            self.ans = dropout(self.ans, training)
            self.ans = tf.reduce_sum(self.ans * self.a_pass, axis=0)

            self.subt = tf.layers.dense(self.raw_subt, hp['emb_dim'] * group, tf.nn.tanh, True, initializer,
                                        kernel_regularizer=reg)
            self.subt = tf.split(self.subt, group, axis=1)
            self.subt = tf.stack(self.subt)
            self.subt = dropout(self.subt, training)
            self.subt = tf.reduce_sum(self.subt * self.s_pass, axis=0)

            if skip:
                self.ques = self.raw_ques + self.ques
                self.ans = self.raw_ans + self.ans
                self.subt = self.raw_subt + self.subt

            if norm:
                self.ques = l2_norm(self.ques)
                self.ans = l2_norm(self.ans)
                self.subt = l2_norm(self.subt)

            if drop:
                self.ques = dropout(self.ques, training)
                self.ans = dropout(self.ans, training)
                self.subt = dropout(self.subt, training)

        with tf.variable_scope('Response'):
            # (N, 1)
            self.sq = tf.matmul(self.subt, self.ques, transpose_b=True)
            self.sq = tf.nn.relu(self.sq)
            # self.sq = tf.nn.softmax(self.sq, 0)
            self.sq = dropout(self.sq, training)
            # self.sq = l2_norm(self.sq, 0)
            # self.sq = l1_norm(self.sq, 0)

            # (N, 5)
            self.sa = tf.matmul(self.subt, self.ans, transpose_b=True)
            self.sa = tf.nn.relu(self.sa)
            # self.sa = tf.nn.softmax(self.sa)
            self.sa = dropout(self.sa, training)
            # self.sa = l2_norm(self.sa, 0)
            # self.sa = l1_norm(self.sa, 0)

            # (N, 5)
            self.attn = self.sq + self.sa
            # self.attn = l1_norm(self.attn, axis=0)
            # (1, N, 5)
            self.attn = tf.expand_dims(self.attn, 0)
            # (5, N, 1)
            self.attn = tf.transpose(self.attn, [2, 1, 0])
            # (5, N, E_t)
            self.abs = tf.expand_dims(self.subt, 0) * self.attn
            # (5, E_t)
            self.abs = tf.reduce_sum(self.abs, axis=1)
            self.abs = l2_norm(self.abs, 1)
            # (5, 1)
            self.output = tf.reduce_sum(self.abs * self.ans, axis=1, keepdims=True)
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

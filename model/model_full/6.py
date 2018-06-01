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
            self.raw_subt = self.data.subt
            self.raw_feat = self.data.feat
            # self.raw_subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
            self.raw_ques = l2_norm(self.raw_ques)
            self.raw_ans = l2_norm(self.raw_ans)
            self.raw_subt = l2_norm(self.raw_subt)
            self.raw_feat = l2_norm(self.raw_feat)

            self.raw_ques = dropout(self.raw_ques, training)
            self.raw_ans = dropout(self.raw_ans, training)
            self.raw_subt = dropout(self.raw_subt, training)
            self.raw_feat = dropout(self.raw_feat, training)

            self.spec = tf.cast(self.data.spec, tf.bool)
            self.neg_spec = tf.logical_not(self.spec)

            self.spec = tf.expand_dims(self.spec, 1)
            self.neg_spec = tf.expand_dims(self.neg_spec, 1)

            self.spec = tf.cast(self.spec, tf.float32)
            self.neg_spec = tf.cast(self.neg_spec, tf.float32)

            # (5, E_t)
            self.ans = tf.layers.dense(self.raw_ans, hp['emb_dim'],
                                       kernel_initializer=init, kernel_regularizer=reg)
            self.ans = self.ans + self.raw_ans
            self.ans = l2_norm(self.ans)
            self.ans = dropout(self.ans, training)

            # (N, E_t)
            # self.subt = tf.layers.dense(self.raw_subt, hp['emb_dim'],  # reuse=True,
            #                             kernel_initializer=init, kernel_regularizer=reg)
            # self.subt = self.subt + self.raw_subt
            # self.subt = l2_norm(self.subt)
            # self.subt = dropout(self.subt, training)

            # (1, E_t)
            self.ques = tf.layers.dense(self.raw_ques, hp['emb_dim'],  # reuse=True,
                                        kernel_initializer=init, kernel_regularizer=reg)
            self.ques = self.ques + self.raw_ques
            self.ques = l2_norm(self.ques)
            self.ques = dropout(self.ques, training)

        with tf.variable_scope('Visual'):
            # (N, 6, E_t)
            self.feat = tf.layers.dense(self.raw_feat, hp['emb_dim'],
                                        kernel_initializer=init, kernel_regularizer=reg)
            self.feat = l2_norm(self.feat)
            self.feat = dropout(self.feat, training)

            # (N, 6, 1)
            self.foq = tf.matmul(self.feat,
                                 tf.tile(tf.expand_dims(self.ques, 0),
                                         [tf.shape(self.feat)[0], 1, 1]),
                                 transpose_b=True)
            self.foq = tf.nn.relu(self.foq)
            self.foq = dropout(self.foq, training)
            # (N, 6, 5)
            self.foa = tf.matmul(self.feat,
                                 tf.tile(tf.expand_dims(self.ans, 0),
                                         [tf.shape(self.feat)[0], 1, 1]),
                                 transpose_b=True)
            self.foa = tf.nn.relu(self.foa)
            self.foa = dropout(self.foa, training)
            # (N, 6, 5)
            self.v_attn = self.foa + self.foq
            # (5, N, 6)
            self.v_attn = tf.transpose(self.v_attn, [2, 0, 1])
            # (5, N, 6, 1)
            self.v_attn = tf.expand_dims(self.v_attn, -1)
            # (5, N, E_t)
            self.visual = tf.reduce_sum(tf.expand_dims(self.feat, 0) * self.v_attn, 2)
            self.visual = l2_norm(self.visual, 2)
            self.visual = dropout(self.visual, training)

            visual = tf.layers.dense(self.visual, hp['emb_dim'],
                                     kernel_initializer=init, kernel_regularizer=reg)
            self.visual = visual + self.visual
            self.visual = l2_norm(self.visual, 2)
            self.visual = dropout(self.visual, training)
            # (5, N, 1)
            self.vq = tf.matmul(self.visual,
                                tf.tile(tf.expand_dims(self.ques, 0), [5, 1, 1]), transpose_b=True)
            self.vq = tf.nn.relu(self.vq)
            self.vq = dropout(self.vq, training)
            # (5, N, 1)
            self.va = tf.matmul(self.visual, tf.expand_dims(self.ans, 1), transpose_b=True)
            self.va = tf.nn.relu(self.va)
            self.va = dropout(self.va, training)
            # (5, E_t)
            self.v_repr = tf.reduce_sum(self.visual * (self.va + self.vq), axis=1)
            self.v_repr = l2_norm(self.v_repr)
            self.v_repr = dropout(self.v_repr, training)

        with tf.variable_scope('Response'):
            # # (N, 1)
            # self.sq = tf.matmul(self.subt, self.ques, transpose_b=True)
            # self.sq = tf.nn.relu(self.sq)
            # self.sq = dropout(self.sq, training)
            # # (N, 5)
            # self.sa = tf.matmul(self.subt, self.ans, transpose_b=True)
            # self.sa = tf.nn.relu(self.sa)
            # self.sa = dropout(self.sa, training)
            # alpha1 = tf.nn.sigmoid(tf.get_variable('alpha1', [], initializer=tf.zeros_initializer()))
            # alpha2 = tf.nn.sigmoid(tf.get_variable('alpha2', [], initializer=tf.zeros_initializer()))
            # # (N, 5)
            # self.attn = self.sq + alpha1 * self.sa
            # self.attn = self.attn * self.belief
            # self.attn = tf.expand_dims(self.attn, 0)
            # self.attn = tf.transpose(self.attn, [2, 1, 0])
            # # (N, 5)
            # self.feat_attn = self.fnq + self.fna * alpha2
            # self.feat_attn = self.feat_attn * self.belief
            # self.feat_attn = tf.expand_dims(self.feat_attn, 0)
            # self.feat_attn = tf.transpose(self.feat_attn, [2, 1, 0])
            # # (1, E_t)
            # self.abst = tf.reduce_sum(self.attn * tf.expand_dims(self.subt, 0), axis=1)
            # self.abst_feat = tf.reduce_sum(self.feat_attn * tf.expand_dims(self.feat_new, 0), axis=1)
            # beta1 = tf.nn.sigmoid(tf.get_variable('beta1', [], initializer=tf.zeros_initializer()))
            # beta2 = tf.nn.sigmoid(tf.get_variable('beta2', [], initializer=tf.zeros_initializer()))
            # self.abst = beta1 * self.abst + self.ques * (1 - beta1) + beta2 * self.abst_feat
            # self.abst = l2_norm(self.abst)
            # self.abst = dropout(self.abst, training)

            self.output = tf.reduce_sum(self.ans * self.v_repr, axis=1, keepdims=True)
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
        a, b = sess.run([model.subt, model.abs])
        print(a, b)
        print(a.shape, b.shape)


if __name__ == '__main__':
    main()

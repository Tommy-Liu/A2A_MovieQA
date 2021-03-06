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

            # self.spec = tf.to_int32(self.spec)
            # self.neg_spec = tf.to_int32(self.neg_spec)

            self.spec = tf.expand_dims(self.spec, 1)
            self.neg_spec = tf.expand_dims(self.neg_spec, 1)

            # self.spec_mask = tf.matmul(self.neg_spec, self.spec, transpose_b=True)
            # self.spec_mask = tf.to_float(self.spec_mask)
            # self.spec_mask = tf.cast(self.spec_mask, tf.bool)
            # self.spec_mask = tf.logical_not(self.spec_mask)
            # self.spec_mask = tf.cast(self.spec_mask, tf.float32)
            # self.spec_mask = self.spec_mask * (-2**32 + 1)

            self.spec = tf.cast(self.spec, tf.float32)
            self.neg_spec = tf.cast(self.neg_spec, tf.float32)

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

            self.feat = tf.layers.dense(self.raw_feat, hp['emb_dim'],
                                        kernel_initializer=init, kernel_regularizer=reg)
            self.feat = l2_norm(self.feat)
            self.feat = dropout(self.feat, training)

            # num_subt = tf.shape(self.subt)[0]

        with tf.variable_scope('Feat_Propagation'):
            # (N, 6, 1)
            self.fq = tf.matmul(self.feat, tf.tile(tf.expand_dims(self.ques, 0), [tf.shape(self.feat)[0], 1, 1]),
                                transpose_b=True)
            self.fq = tf.nn.relu(self.fq)
            self.fq = dropout(self.fq, training)

            # (N, E_t)
            self.feat_new = tf.reduce_sum(self.feat * self.fq, axis=1)
            self.feat_new = l2_norm(self.feat_new)
            self.feat_new = dropout(self.feat_new, training)

            self.mut_feat = tf.layers.dense(self.feat_new + self.ques, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_feat = l2_norm(self.mut_feat)
            self.mut_feat = dropout(self.mut_feat, training)
            self.mut_feat = tf.layers.dense(self.mut_feat, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_feat = l2_norm(self.mut_feat)
            self.mut_feat = dropout(self.mut_feat, training)
            self.mut_feat = tf.layers.dense(self.mut_feat, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_feat = l2_norm(self.mut_feat)
            self.mut_feat = dropout(self.mut_feat, training)
            self.mut_feat = tf.layers.dense(self.mut_feat, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_feat = l2_norm(self.mut_feat)
            self.mut_feat = dropout(self.mut_feat, training)
            self.mut_feat = tf.layers.dense(self.mut_feat, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_feat = l2_norm(self.mut_feat)
            self.mut_feat = dropout(self.mut_feat, training)
            # (N_s, E_t)
            self.spec_feat = tf.boolean_mask(self.feat_new, tf.cast(self.data.spec, tf.bool))
            # (N, N_s)
            self.feat_prop = tf.matmul(self.mut_feat, self.spec_feat, transpose_b=True)
            self.feat_prop = tf.nn.relu(self.feat_prop)
            # (N, 1)
            self.feat_prop = tf.reduce_mean(self.feat_prop, axis=1, keepdims=True)
            # (N, 1)
            self.feat_belief = self.feat_prop * self.neg_spec
            # max_b = tf.maximum(tf.reduce_max(self.belief), 10 ** (-6))
            # self.belief = self.belief / max_b
            self.feat_belief = dropout(self.feat_belief, training)
            # self.belief = tf.reduce_max(self.propagation, axis=1, keepdims=True)
            self.feat_belief = self.spec + self.feat_belief
            self.feat_belief = tf.minimum(self.feat_belief, 1.0)

        with tf.variable_scope('Subt_Propagation'):
            self.mut_subt = tf.layers.dense(self.subt + self.ques, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_subt = l2_norm(self.mut_subt)
            self.mut_subt = dropout(self.mut_subt, training)
            self.mut_subt = tf.layers.dense(self.mut_subt, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_subt = l2_norm(self.mut_subt)
            self.mut_subt = dropout(self.mut_subt, training)
            self.mut_subt = tf.layers.dense(self.mut_subt, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_subt = l2_norm(self.mut_subt)
            self.mut_subt = dropout(self.mut_subt, training)
            self.mut_subt = tf.layers.dense(self.mut_subt, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_subt = l2_norm(self.mut_subt)
            self.mut_subt = dropout(self.mut_subt, training)
            self.mut_subt = tf.layers.dense(self.mut_subt, hp['emb_dim'],
                                            kernel_initializer=init, kernel_regularizer=reg)
            self.mut_subt = l2_norm(self.mut_subt)
            self.mut_subt = dropout(self.mut_subt, training)
            # (N_s, E_t)
            self.spec_subt = tf.boolean_mask(self.subt, tf.cast(self.data.spec, tf.bool))
            # (N, N_s)
            self.subt_prop = tf.matmul(self.mut_subt, self.spec_subt, transpose_b=True)
            self.subt_prop = tf.nn.relu(self.subt_prop)
            # (N, 1)
            self.subt_prop = tf.reduce_mean(self.subt_prop, axis=1, keepdims=True)
            # (N, 1)
            self.subt_belief = self.subt_prop * self.neg_spec
            # max_b = tf.maximum(tf.reduce_max(self.belief), 10 ** (-6))
            # self.belief = self.belief / max_b
            self.subt_belief = dropout(self.subt_belief, training)
            # self.belief = tf.reduce_max(self.propagation, axis=1, keepdims=True)
            self.subt_belief = self.spec + self.subt_belief
            self.subt_belief = tf.minimum(self.subt_belief, 1.0)

        with tf.variable_scope('Response'):
            # (N, 5)
            self.fna = tf.matmul(self.feat_new, self.ans, transpose_b=True)
            self.fna = tf.nn.relu(self.fna)
            self.fna = dropout(self.fna, training)
            # (N, 1)
            self.fnq = tf.matmul(self.feat_new, self.ques, transpose_b=True)
            self.fnq = tf.nn.relu(self.fnq)
            self.fnq = dropout(self.fnq, training)

            # (N, 1)
            self.sq = tf.matmul(self.subt, self.ques, transpose_b=True)
            self.sq = tf.nn.relu(self.sq)
            self.sq = dropout(self.sq, training)
            # (N, 5)
            self.sa = tf.matmul(self.subt, self.ans, transpose_b=True)
            self.sa = tf.nn.relu(self.sa)
            self.sa = dropout(self.sa, training)
            alpha1 = tf.nn.sigmoid(tf.get_variable('alpha1', [], initializer=tf.zeros_initializer()))
            alpha2 = tf.nn.sigmoid(tf.get_variable('alpha2', [], initializer=tf.zeros_initializer()))
            # (N, 5)
            self.attn = self.sq + alpha1 * self.sa
            self.attn = self.attn * self.subt_belief
            self.attn = tf.expand_dims(self.attn, 0)
            self.attn = tf.transpose(self.attn, [2, 1, 0])
            # (N, 5)
            self.feat_attn = self.fnq + self.fna * alpha2
            self.feat_attn = self.feat_attn * self.feat_belief
            self.feat_attn = tf.expand_dims(self.feat_attn, 0)
            self.feat_attn = tf.transpose(self.feat_attn, [2, 1, 0])
            # (1, E_t)
            self.abst = tf.reduce_sum(self.attn * tf.expand_dims(self.subt, 0), axis=1)
            self.abst_feat = tf.reduce_sum(self.feat_attn * tf.expand_dims(self.feat_new, 0), axis=1)
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
        a, b = sess.run([model.subt, model.abs])
        print(a, b)
        print(a.shape, b.shape)


if __name__ == '__main__':
    main()

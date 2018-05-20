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


class Model(object):
    def __init__(self, data, beta=0.0, training=False):
        self.data = data
        reg = layers.l2_regularizer(beta)
        initializer = tf.glorot_normal_initializer(seed=0)

        def dense(x, units=hp['emb_dim'], activation=None, reuse=False, drop=True, use_bias=True, norm=True,
                  residual=False):
            xx = x
            x = tf.layers.dense(x, units, activation, use_bias=use_bias,  # kernel_constraint=constraint,
                                kernel_initializer=initializer, kernel_regularizer=reg,
                                reuse=reuse)
            if residual:
                x = x + xx
            if norm:
                x = l2_norm(x)
            if drop:
                x = dropout(x, training)
            return x

        with tf.variable_scope('Embedding_Linear'):
            self.raw_ques = self.data.ques
            self.raw_ans = self.data.ans
            self.raw_subt = self.data.subt
            # self.raw_subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
            self.raw_ques = l2_norm(self.raw_ques)
            self.raw_ans = l2_norm(self.raw_ans)
            self.raw_subt = l2_norm(self.raw_subt)
            self.spec = tf.cast(self.data.spec, tf.bool)

            self.ques = self.raw_ques
            self.ans = self.raw_ans
            self.subt = self.raw_subt

            # (5, E_t)
            # self.ans = dense(self.raw_ans, hp['emb_dim'],
            #                  # tf.nn.tanh,
            #                  # norm=False,
            #                  # drop=False,
            #                  residual=True
            #                  )
            #
            # # (N, E_t)
            # self.subt = dense(self.raw_subt, hp['emb_dim'],
            #                   # tf.nn.tanh,
            #                   # norm=False,
            #                   # drop=False,
            #                   residual=True,
            #                   # reuse=True
            #                   )
            # # (1, E_t)
            # self.ques = dense(self.raw_ques, hp['emb_dim'],
            #                   # tf.nn.tanh,
            #                   # norm=False,
            #                   # drop=False,
            #                   residual=True
            #                   )

        with tf.variable_scope('Response'):
            # w = tf.get_variable(_WEIGHTS_VARIABLE_NAME, shape=[2 * hp['emb_dim'], 1], initializer=initializer)
            # *_, self.abs = tf.while_loop(cond, body, [0, self.subt, self.ques, w, tf.zeros_like(self.ques)])
            # self.trail, _= tf.scan(scan_fn, self.subt, (self.ques, w))
            # self.abs = self.trail[-1]

            # self.clue = tf.concat([self.subt, tf.tile(self.ques, [tf.shape(self.subt)[0], 1])], axis=1)
            # (N, E_t)
            self.clue_head = tf.layers.dense(self.subt, hp['emb_dim'], activation=tf.nn.tanh,
                                             kernel_initializer=initializer, kernel_regularizer=reg)
            self.clue_gate = tf.layers.dense(self.subt, hp['emb_dim'], activation=tf.nn.sigmoid,
                                             kernel_initializer=initializer, kernel_regularizer=reg)
            self.clue = self.clue_head * self.clue_gate
            self.clue = l2_norm(self.clue)
            self.clue = dropout(self.clue, training)
            # (N, 1)
            self.mask = tf.matmul(self.clue, self.ques, transpose_b=True)
            self.mask = tf.nn.relu(self.mask)
            # (N, 1)
            # self.mask = dropout(self.mask, training)

            # (N, E_t)
            self.ext_subt = self.ques * self.mask + self.subt

            # (N, E_t)
            self.sa_head = tf.layers.dense(self.ext_subt, hp['emb_dim'], activation=tf.nn.tanh,
                                           kernel_initializer=initializer, kernel_regularizer=reg)
            self.sa_gate = tf.layers.dense(self.ext_subt, hp['emb_dim'], activation=tf.nn.sigmoid,
                                           kernel_initializer=initializer, kernel_regularizer=reg)
            self.sa = self.sa_head * self.sa_gate
            self.sa = l2_norm(self.sa)
            self.sa = dropout(self.sa, training)

            self.attn = tf.matmul(self.sa, self.ans, transpose_b=True)
            self.attn = tf.nn.relu(self.attn)

            # self.attn = dropout(self.attn, training)

            # (1, N, 5)
            self.attn = tf.expand_dims(self.attn, 0)
            # (5, N, 1)
            self.attn = tf.transpose(self.attn, [2, 1, 0])
            # (5, N, E_t)
            self.abs = tf.expand_dims(self.ext_subt, 0) * self.attn
            self.abs = tf.reduce_sum(self.abs, axis=1)
            self.abs = l2_norm(self.abs)
            self.output = tf.reduce_sum(self.abs * self.ans, axis=1, keepdims=True)
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

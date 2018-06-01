import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from raw_input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512, 'dropout_rate': 0.3}

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
            self.raw_subt = tf.boolean_mask(self.data.subt, tf.cast(self.data.spec, tf.bool))
            self.raw_ques = l2_norm(self.raw_ques)
            self.raw_ans = l2_norm(self.raw_ans)
            self.raw_subt = l2_norm(self.raw_subt)

            self.raw_ques = dropout(self.raw_ques, training)
            self.raw_ans = dropout(self.raw_ans, training)
            self.raw_subt = dropout(self.raw_subt, training)

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

            # num_subt = tf.shape(self.subt)[0]

        with tf.variable_scope('Response'):
            split = 30
            dim = 10
            # (N, E_t / 3, 3)
            self.quant_subt = tf.reshape(self.subt, [-1, split, dim])
            self.quant_subt = l2_norm(self.quant_subt, 2)
            # (5, E_t / 3, 3)
            self.quant_ans = tf.reshape(self.ans, [5, split, dim])
            self.quant_ans = l2_norm(self.quant_ans, 2)
            # (1, E_t / 3, 3)
            self.quant_ques = tf.reshape(self.ques, [1, split, dim])
            self.quant_ques = l2_norm(self.quant_ques, 2)
            # (E_t / 3, 3, 3, 3)
            self.tri_matrix = tf.get_variable('tri_matrix', [split, dim, dim, dim],
                                              initializer=init, regularizer=reg)
            # (N, E_t / 3, 3, 3)
            self.first = tf.einsum('ijk,jklm->ijlm', self.quant_subt, self.tri_matrix)
            # (5, N, E_t / 3, 3)
            self.second = tf.einsum('ijk,hjkl->ihjl', self.quant_ans, self.first)
            # (5, N, E_t / 3, 1)
            self.third = tf.einsum('ijk,ghjk->ghji', self.quant_ques, self.second)

            self.third = tf.nn.relu(self.third)
            self.third = dropout(self.third, training)

            self.abs = tf.expand_dims(self.quant_subt, 0) * self.third
            self.abs = tf.reshape(self.abs, [5, -1, hp['emb_dim']])
            self.abs = tf.reduce_sum(self.abs, axis=1)
            self.abs = l2_norm(self.abs)

            self.output = tf.reduce_sum(self.ans * self.abs, axis=1, keepdims=True)
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
        a, b = sess.run([model.third, model.second])
        print(a, b)
        print(a.shape, b.shape)
        # print(np.array_equal(a[0, :3], np.squeeze(b[0, 0, :])))


if __name__ == '__main__':
    main()

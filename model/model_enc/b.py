import numpy as np
import tensorflow as tf

from config import MovieQAPath
from legacy.input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 300, 'feat_dim': 512,
      'learning_rate': 10 ** (-3), 'decay_rate': 0.83, 'decay_type': 'exp', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1, 'pos_len': 40}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def language_encode(a, b, al, bl, a_pos, b_pos):
    gamma = tf.convert_to_tensor(10 ** (-8), tf.float32)

    al, bl = tf.expand_dims(al, axis=-1), tf.expand_dims(bl, axis=-1)

    ab_dot = tf.tensordot(a, b, axes=[[-1], [-1]]) / (hp['emb_dim'] ** 0.5)  # (1, L_q, N, L_s)

    a_attention = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(ab_dot, axis=[2, 3])), axis=2)  # (1, L_q, 1)

    a_output = (a + a * a_attention) * tf.nn.tanh(a_pos)  # (1, L_q, E_t)

    a_enc = tf.reduce_sum(a_output, axis=1) / (tf.to_float(al) + gamma)  # (1, E_t)

    b_attention = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(ab_dot, axis=[0, 1])), axis=2)  # (N, L_s)

    b_output = (b + b * b_attention) * tf.nn.tanh(b_pos)  # (N, L_s, E_t)

    b_enc = tf.reduce_sum(b_output, axis=1) / (tf.to_float(bl) + gamma)  # (N, E_t)

    return a_enc, b_enc


class Model(object):
    def __init__(self, data, training=False):
        self.data = data
        self.initializer = tf.orthogonal_initializer()
        q_mask = tf.sequence_mask(self.data.ql, maxlen=25)  # (1, L_q)
        s_mask = tf.sequence_mask(self.data.sl, maxlen=29)  # (N, L_s)
        a_mask = tf.sequence_mask(self.data.al, maxlen=34)  # (5, L_a)

        with tf.variable_scope('Embedding'):
            self.embedding = tf.get_variable('embedding_matrix',
                                             initializer=np.load(_mp.embedding_file), trainable=False)

            self.ques = tf.nn.embedding_lookup(self.embedding, self.data.ques)  # (1, L_q, E)
            self.ans = tf.nn.embedding_lookup(self.embedding, self.data.ans)  # (5, L_a, E)
            self.subt = tf.nn.embedding_lookup(self.embedding, self.data.subt)  # (N, L_s, E)

            # self.ques = tf.layers.dropout(self.ques, hp['dropout_rate'], training=training)  # (1, L_q, E)
            # self.ans = tf.layers.dropout(self.ans, hp['dropout_rate'], training=training)  # (5, L_a, E)
            # self.subt = tf.layers.dropout(self.subt, hp['dropout_rate'], training=training)  # (N, L_s, E)

        with tf.variable_scope('Embedding_Linear'):
            self.ques_embedding = tf.layers.dense(
                self.ques, hp['emb_dim'], use_bias=False, kernel_initializer=self.initializer)  # (1, L_q, E_t)
            self.ans_embedding = tf.layers.dense(self.ans, hp['emb_dim'], use_bias=False, reuse=True)  # (5, L_a, E_t)
            self.subt_embedding = tf.layers.dense(self.subt, hp['emb_dim'], use_bias=False,
                                                  reuse=True, )  # (N, L_s, E_t)

        with tf.variable_scope('Language_Encode'):
            position_attn = tf.get_variable('position_attention', shape=[hp['pos_len'], hp['emb_dim']],
                                            initializer=self.initializer, trainable=False)
            ques_pos, _ = tf.split(position_attn, [25, hp['pos_len'] - 25])
            ans_pos, _ = tf.split(position_attn, [34, hp['pos_len'] - 34])
            subt_pos, _ = tf.split(position_attn, [29, hp['pos_len'] - 29])

            q_qa_enc, a_qa_enc = language_encode(self.ques, self.ans, self.data.ql, self.data.al, ques_pos, ans_pos)
            q_qs_enc, s_qs_enc = language_encode(self.ques, self.subt, self.data.ql, self.data.sl, ques_pos, subt_pos)
            a_as_enc, s_as_enc = language_encode(self.ans, self.subt, self.data.al, self.data.sl, ans_pos, subt_pos)

            self.ques_enc = tf.layers.dense(tf.concat(
                [q_qa_enc, q_qs_enc], axis=-1), hp['feat_dim'],
                kernel_initializer=self.initializer, activation=tf.nn.tanh)  # (1, L_q, 2 * E_t)
            self.ans_enc = tf.layers.dense(tf.concat(
                [a_qa_enc, a_as_enc], axis=-1), hp['feat_dim'],
                kernel_initializer=self.initializer, activation=tf.nn.tanh)  # (5, L_a, 2 * E_t)
            self.subt_enc = tf.layers.dense(tf.concat(
                [s_qs_enc, s_as_enc], axis=-1), hp['feat_dim'],
                kernel_initializer=self.initializer, activation=tf.nn.tanh)  # (N, L_s, 2 * E_t)

        #
        #     self.ques_enc = tf.layers.dense(self.ques_enc, hp['feat_dim'])  # (1, L_q, 2 * E_t)
        #     self.ans_enc = tf.layers.dense(self.ques_enc, hp['feat_dim'])  # (5, L_a, 2 * E_t)
        #     self.subt_enc = tf.layers.dense(self.ques_enc, hp['feat_dim'])  # (N, L_s, 2 * E_t)
        #
        # self.m_subt = tf.layers.dense(
        #     self.subt_enc, hp['feat_dim'], use_bias=False, name='encode_transform')  # (N, F_t)
        # self.m_ques = tf.layers.dense(
        #     self.ques_enc, hp['feat_dim'], use_bias=False, reuse=True, name='encode_transform')  # (1, F_t)
        # self.m_ans = tf.layers.dense(
        #     self.ans_enc, hp['feat_dim'], use_bias=False, reuse=True, name='encode_transform')  # (5, F_t)
        #
        # self.m_subt = tf.layers.dropout(self.m_subt, hp['dropout_rate'], training=training)
        # self.m_ques = tf.layers.dropout(self.m_ques, hp['dropout_rate'], training=training)
        # self.m_ans = tf.layers.dropout(self.m_ans, hp['dropout_rate'], training=training)
        #
        t_shape = tf.shape(self.subt_enc)
        split_num = tf.cast(tf.ceil(t_shape[0] / 5), dtype=tf.int32)
        pad_num = split_num * 5 - t_shape[0]
        paddings = tf.convert_to_tensor([[0, pad_num], [0, 0]])

        with tf.variable_scope('Memory_Block'):
            self.mem_feat = tf.pad(self.subt_enc, paddings)

            self.mem_block = tf.reshape(self.mem_feat, [split_num, 5, hp['feat_dim']])

            self.mem_node = tf.reduce_mean(self.mem_block, axis=1)

            self.mem_opt = tf.layers.dense(self.mem_node, hp['feat_dim'],
                                           activation=tf.nn.tanh, kernel_initializer=self.initializer)

            self.mem_direct = tf.matmul(self.mem_node, self.mem_opt, transpose_b=True) / (hp['feat_dim'] ** 0.5)

            self.mem_fw_direct = tf.nn.softmax(self.mem_direct)

            self.mem_bw_direct = tf.nn.softmax(self.mem_direct, axis=0)

            self.mem_self = tf.matmul(self.mem_fw_direct, self.mem_node) + tf.matmul(self.mem_bw_direct, self.mem_node)

        self.mem_attn = tf.nn.softmax(tf.matmul(self.mem_self, self.ques_enc, transpose_b=True))

        self.mem_output = tf.reduce_sum(self.mem_self * self.mem_attn, axis=0)

        self.output = tf.reduce_sum(self.mem_output * self.ans_enc, axis=1)


def main():
    data = Input(split='train')
    model = Model(data)

    for v in tf.global_variables():
        print(v)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run([model.data.initializer, tf.global_variables_initializer()], )

        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        t, tt = sess.run([model.output, data.gt])
        print(t, tt)
        print(t.shape, tt.shape)


if __name__ == '__main__':
    main()

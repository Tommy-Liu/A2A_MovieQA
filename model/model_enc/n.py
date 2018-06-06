import numpy as np
import tensorflow as tf

from config import MovieQAPath
from legacy.input import Input

_mp = MovieQAPath()
hp = {'emb_lin_dim': 300, 'feat_dim': 512,
      'learning_rate': 10 ** (-4), 'decay_rate': 0.5, 'decay_type': 'exp', 'decay_epoch': 1,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def language_encode(a, b, mask_a, mask_b, al, bl, reuse=False, training=True):
    gamma = tf.convert_to_tensor(10 ** (-8), tf.float32)

    al = tf.expand_dims(al, axis=-1)
    bl = tf.expand_dims(bl, axis=-1)

    ab_value = tf.tensordot(mask_a, mask_b, axes=[[-1], [-1]])  # (1, L_q, N, L_s)
    ab_value = ab_value / (mask_a.get_shape().as_list()[-1] ** 0.5)  # (1, L_q, N, L_s)

    a_value = tf.reduce_sum(ab_value, axis=[2, 3])  # (1, L_q)

    a_attention = tf.expand_dims(tf.nn.softmax(a_value), axis=2)  # (1, L_q, 1)

    a_output = dropout(mask_a * a_attention, training=training)  # (1, L_q, E_t)

    a_attended = tf.contrib.layers.layer_norm(
        a_output + a, begin_norm_axis=2, reuse=reuse, scope=tf.get_variable_scope())  # (1, L_q, E_t)

    a_enc = tf.reduce_sum(a_attended, axis=1) / (tf.to_float(al) + gamma)  # (1, E_t)

    b_value = tf.reduce_sum(ab_value, axis=[0, 1])  # (N, L_s)

    b_attention = tf.expand_dims(tf.nn.softmax(b_value), axis=2)  # (N, L_s)

    b_output = dropout(mask_b * b_attention, training=training)  # (N, L_s, E_t)

    b_attended = tf.contrib.layers.layer_norm(
        b_output + b, begin_norm_axis=2, reuse=True, scope=tf.get_variable_scope())  # (N, L_s, E_t)

    b_enc = tf.reduce_sum(b_attended, axis=1) / (tf.to_float(bl) + gamma)  # (N, E_t)

    return a_enc, b_enc


class Model(object):
    def __init__(self, data, training=True):
        self.data = data

        with tf.variable_scope('Embedding'):
            self.embedding = tf.get_variable(
                'embedding_matrix', initializer=np.load(_mp.embedding_file), trainable=False)

            self.ques = tf.nn.embedding_lookup(self.embedding, self.data.ques)  # (1, L_q, E)
            self.ans = tf.nn.embedding_lookup(self.embedding, self.data.ans)  # (5, L_a, E)
            self.subt = tf.nn.embedding_lookup(self.embedding, self.data.subt)  # (N, L_s, E)

            self.ques = tf.layers.dropout(self.ques, hp['dropout_rate'], training=training)  # (1, L_q, E)
            self.ans = tf.layers.dropout(self.ans, hp['dropout_rate'], training=training)  # (5, L_a, E)
            self.subt = tf.layers.dropout(self.subt, hp['dropout_rate'], training=training)  # (N, L_s, E)

        q_mask = tf.sequence_mask(self.data.ql, maxlen=25)  # (1, L_q)
        s_mask = tf.sequence_mask(self.data.sl, maxlen=29)  # (N, L_s)
        a_mask = tf.sequence_mask(self.data.al, maxlen=34)  # (5, L_a)
        ques_shape = self.ques.get_shape().as_list()
        subt_shape = self.subt.get_shape().as_list()
        ans_shape = self.ans.get_shape().as_list()

        with tf.variable_scope('Embedding_Linear'):
            self.ques_embedding = tf.layers.dense(self.ques, hp['emb_lin_dim'], name='embed_transform')  # (1, L_q, E_t)
            self.ans_embedding = tf.layers.dense(
                self.ans, hp['emb_lin_dim'], reuse=True, name='embed_transform')  # (5, L_a, E_t)
            self.subt_embedding = tf.layers.dense(
                self.subt, hp['emb_lin_dim'], reuse=True, name='embed_transform')  # (N, L_s, E_t)

            q_padding = tf.zeros_like(self.ques_embedding)  # (1, L_q, E_t)
            q_emb_mask = tf.tile(tf.expand_dims(q_mask, axis=-1), [1, 1, ques_shape[-1]])  # (1, L_q, E_t)

            self.mask_q_emb = tf.where(q_emb_mask, self.ques_embedding, q_padding)  # (1, L_q, E_t)

            s_padding = tf.zeros_like(self.subt_embedding)  # (N, L_s, E_t)
            s_emb_mask = tf.tile(tf.expand_dims(s_mask, axis=-1), [1, 1, subt_shape[-1]])  # (N, L_s, E_t)

            self.mask_s_emb = tf.where(s_emb_mask, self.subt_embedding, s_padding)  # (N, L_s, E_t)

            a_padding = tf.zeros_like(self.ans_embedding)  # (5, L_a, E_t)
            a_emb_mask = tf.tile(tf.expand_dims(a_mask, axis=-1), [1, 1, ans_shape[-1]])  # (5, L_a, E_t)

            self.mask_a_emb = tf.where(a_emb_mask, self.ans_embedding, a_padding)  # (5, L_a, E_t)

        with tf.variable_scope('Language_Encode'):
            q_qa_enc, a_qa_enc = language_encode(self.ques, self.ans, self.mask_q_emb, self.mask_a_emb, self.data.ql,
                                                 self.data.al, training=training)
            q_qs_enc, s_qs_enc = language_encode(self.ques, self.subt, self.mask_q_emb, self.mask_s_emb, self.data.ql,
                                                 self.data.sl, reuse=True, training=training)
            a_as_enc, s_as_enc = language_encode(self.ans, self.subt, self.mask_a_emb, self.mask_s_emb, self.data.al,
                                                 self.data.sl, reuse=True, training=training)

            self.ques_enc = tf.layers.dense(tf.concat(
                [q_qa_enc, q_qs_enc], axis=-1), hp['feat_dim'], activation=tf.nn.relu)  # (1, L_q, 2 * E_t)
            self.ans_enc = tf.layers.dense(tf.concat(
                [a_qa_enc, a_as_enc], axis=-1), hp['feat_dim'], activation=tf.nn.relu)  # (5, L_a, 2 * E_t)
            self.subt_enc = tf.layers.dense(tf.concat(
                [s_qs_enc, s_as_enc], axis=-1), hp['feat_dim'], activation=tf.nn.relu)  # (N, L_s, 2 * E_t)

            self.ques_enc = tf.layers.dense(self.ques_enc, hp['feat_dim'])  # (1, L_q, 2 * E_t)
            self.ans_enc = tf.layers.dense(self.ques_enc, hp['feat_dim'])  # (5, L_a, 2 * E_t)
            self.subt_enc = tf.layers.dense(self.ques_enc, hp['feat_dim'])  # (N, L_s, 2 * E_t)

            # self.qs_value = tf.tensordot(self.mask_q_emb, self.mask_s_emb, axes=[[-1], [-1]])  # (1, L_q, N, L_s)
            # self.qs_value = self.qs_value / (self.mask_q_emb.get_shape().as_list()[-1] ** 0.5)  # (1, L_q, N, L_s)
            #
            # self.q_value = tf.reduce_sum(self.qs_value, axis=[2, 3])  # (1, L_q)
            #
            # self.q_attention = tf.expand_dims(tf.nn.softmax(self.q_value), axis=2)  # (1, L_q, 1)
            #
            # self.ques_attended = tf.contrib.layers.layer_norm(
            #     self.mask_q_emb * self.q_attention + self.ques, begin_norm_axis=2)  # (1, L_q, E_t)
            #
            # self.ques_enc = tf.reduce_sum(
            #     self.ques_attended, axis=1) / tf.sqrt(tf.to_float(self.data.ql))  # (1, E_t)
            #
            # self.s_value = tf.reduce_sum(self.qs_value, axis=[0, 1])  # (N, L_s)
            #
            # self.s_attention = tf.expand_dims(tf.nn.softmax(self.s_value), axis=2)  # (N, L_s)
            #
            # self.subt_attended = tf.contrib.layers.layer_norm(
            #     self.mask_s_emb * self.s_attention + self.subt, begin_norm_axis=2)  # (N, L_s, E_t)
            #
            # self.subt_enc = tf.reduce_sum(self.mask_s_emb * self.s_attention, axis=1) \
            #                 / tf.sqrt(tf.to_float(self.data.sl))  # (N, E_t)

        self.feat = tf.layers.dense(self.data.feat, hp['feat_dim'], name='feature_transform')  # (N, 64, F_t)
        self.m_subt = tf.layers.dense(
            self.subt_enc, hp['feat_dim'], use_bias=False, name='encode_transform')  # (N, F_t)
        self.m_ques = tf.layers.dense(
            self.ques_enc, hp['feat_dim'], use_bias=False, reuse=True, name='encode_transform')  # (1, F_t)
        self.m_ans = tf.layers.dense(
            self.ans_enc, hp['feat_dim'], use_bias=False, reuse=True, name='encode_transform')  # (5, F_t)

        self.m_subt = tf.layers.dropout(self.m_subt, hp['dropout_rate'], training=training)
        self.m_ques = tf.layers.dropout(self.m_ques, hp['dropout_rate'], training=training)
        self.m_ans = tf.layers.dropout(self.m_ans, hp['dropout_rate'], training=training)

        with tf.variable_scope('Spatial_Attention'):
            self.qs_aware = tf.nn.tanh(self.m_subt + tf.tile(self.m_ques, [tf.shape(self.m_subt)[0], 1]))  # (N, F_t)

            self.position_matrix = tf.get_variable('position_matrix', shape=[1, 64, hp['feat_dim']])  # (1, 64, F_t)

            self.spatial_attention = tf.expand_dims(tf.nn.softmax(
                tf.einsum('ijk,ik->ij', self.feat + self.position_matrix, self.qs_aware)), axis=2)  # (N, 64)

            self.feat_weighted_sum = tf.reduce_sum(self.feat * self.spatial_attention, axis=1)  # (N, F_t)

            self.feat_weighted_sum = tf.layers.dropout(self.feat_weighted_sum, hp['dropout_rate'], training=training)

        with tf.variable_scope('Co_Attention'):
            self.fs_aware = tf.nn.tanh(tf.layers.dense(self.feat_weighted_sum, hp['feat_dim'], use_bias=False) +
                                       self.m_subt + tf.tile(self.m_ques, [tf.shape(self.m_subt)[0], 1]))  # (N, F_t)

            self.f_co_attention = tf.expand_dims(tf.nn.softmax(
                tf.einsum('ij,ij->i', self.feat_weighted_sum, self.fs_aware)), axis=1)  # (N, 1)

            self.s_co_attention = tf.expand_dims(tf.nn.softmax(
                tf.einsum('ij,ij->i', self.m_subt, self.fs_aware)), axis=1)  # (N, 1)

            self.f_co_attended = tf.layers.dropout(self.f_co_attention * self.feat_weighted_sum,
                                                   hp['dropout_rate'], training=training)  # (N, F_t)
            self.s_co_attended = tf.layers.dropout(self.s_co_attention * self.m_subt,
                                                   hp['dropout_rate'], training=training)  # (N, F_t)

            self.story_feat = self.f_co_attended + self.s_co_attended  # (N, F_t)

        t_shape = tf.shape(self.story_feat)
        split_num = tf.cast(tf.ceil(t_shape[0] / 5), dtype=tf.int32)
        pad_num = split_num * 5 - t_shape[0]
        paddings = tf.convert_to_tensor([[0, pad_num], [0, 0]])

        with tf.variable_scope('Memory_Block'):
            self.mem_feat = tf.pad(self.story_feat, paddings)

            self.mem_block = tf.reshape(self.mem_feat, [split_num, 5, hp['feat_dim']])

            self.mem_node = tf.reduce_mean(self.mem_block, axis=1)

            self.mem_opt = tf.layers.dense(self.mem_node, hp['feat_dim'], use_bias=False)

            self.mem_direct = tf.nn.softmax(tf.matmul(
                self.mem_node, self.mem_opt, transpose_b=True) /
                                            (self.mem_node.get_shape().as_list()[-1] ** 0.5))

            self.mem_ans = tf.reduce_mean(tf.matmul(self.mem_direct, self.mem_node), axis=0, keepdims=True)

        self.output = tf.reduce_sum(self.mem_ans * self.m_ans, axis=1)


def main():
    data = Input(split='train')
    model = Model(data)

    for v in tf.global_variables():
        print(v)
    with tf.Session() as sess:
        sess.run([model.data.initializer, tf.global_variables_initializer()], )

        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        t, tt = sess.run([model.output, data.gt])
        print(t, tt)
        print(t.shape, tt.shape)


if __name__ == '__main__':
    main()

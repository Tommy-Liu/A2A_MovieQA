import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import AttentionCellWrapper, CompiledWrapper

from config import MovieQAPath
from input import Input

_mp = MovieQAPath()

hp = {'emb_lin_dim': 300, 'feat_dim': 600,
      'learning_rate': 10 ** (-3), 'decay_rate': 0.83, 'decay_type': 'exp', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1, 'attn_length': 10}


def get_cell():
    cell = tf.nn.rnn_cell.GRUCell(hp['emb_lin_dim'], kernel_initializer=tf.orthogonal_initializer())
    cell = AttentionCellWrapper(cell, hp['attn_length'])
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1 - hp['dropout_rate'])
    cell = CompiledWrapper(cell)

    return cell


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

            self.ques_embedding = self.ques + tf.layers.dense(
                self.ques, hp['emb_lin_dim'], activation=tf.nn.relu, name='Embed_Linear')  # (1, L_q, E_t)
            self.ans_embedding = self.ans + tf.layers.dense(
                self.ans, hp['emb_lin_dim'], activation=tf.nn.relu, reuse=True, name='Embed_Linear')  # (5, L_a, E_t)
            self.subt_embedding = self.subt + tf.layers.dense(
                self.subt, hp['emb_lin_dim'], activation=tf.nn.relu, reuse=True, name='Embed_Linear')  # (N, L_s, E_t)

        with tf.variable_scope('Language_Encode'):
            fw_cell, bw_cell = get_cell(), get_cell()  # (state (E_t), attn(E_t), attn_state (E_t * attn_length))

            self.ques_output, self.ques_states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.ques_embedding, self.data.ql, dtype=tf.float32)

            self.ans_output, self.ans_states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.ans_embedding, self.data.al, dtype=tf.float32)

            self.subt_output, self.subt_states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.subt_embedding, self.data.sl, dtype=tf.float32)

            self.ques_enc = tf.concat([self.ques_states[0][0], self.ques_states[1][0]], axis=1)
            self.ans_enc = tf.concat([self.ans_states[0][0], self.ans_states[1][0]], axis=1)
            self.subt_enc = tf.concat([self.subt_states[0][0], self.subt_states[1][0]], axis=1)

            self.ques_enc = self.ques_enc + \
                            tf.layers.dense(self.ques_enc, hp['feat_dim'], activation=tf.nn.relu)  # (1, 2 * E_t)
            self.ans_enc = self.ans_enc + \
                           tf.layers.dense(self.ques_enc, hp['feat_dim'], activation=tf.nn.relu)  # (5, 2 * E_t)
            self.subt_enc = self.subt_enc + \
                            tf.layers.dense(self.ques_enc, hp['feat_dim'], activation=tf.nn.relu)  # (N, 2 * E_t)

        with tf.variable_scope('Spatial_Attention'):
            self.position_matrix = tf.get_variable('position_matrix', shape=[1, 64, hp['feat_dim']],
                                                   initializer=tf.orthogonal_initializer())  # (1, 64, F_t)
            self.feat = tf.layers.dense(self.data.feat, hp['feat_dim'], activation=tf.nn.relu) + \
                        self.position_matrix  # (N, 64, F_t)

            self.qs_aware = tf.nn.tanh(
                self.subt_enc + tf.tile(self.ques_enc, [tf.shape(self.subt_enc)[0], 1]))  # (N, F_t)

            self.spatial_attention = tf.expand_dims(tf.nn.softmax(tf.einsum(
                'ijk,ik->ij', self.feat, self.qs_aware) / (hp['feat_dim'] ** 0.5)), axis=2)  # (N, 64)

            self.feat_weighted_sum = tf.reduce_sum(self.feat * self.spatial_attention + self.feat, axis=1)  # (N, F_t)

            self.vis_enc = tf.layers.dense(self.feat_weighted_sum, hp['feat_dim'], activation=tf.nn.relu)  # (N, F_t)

        self.feat_concat = self.vis_enc + self.subt_enc  # (N, F_t)

        t_shape = tf.shape(self.feat_concat)
        split_num = tf.cast(tf.ceil(t_shape[0] / 5), dtype=tf.int32)
        pad_num = split_num * 5 - t_shape[0]
        paddings = tf.convert_to_tensor([[0, pad_num], [0, 0]])

        with tf.variable_scope('Memory_Block'):
            self.mem_feat = tf.pad(self.feat_concat, paddings)

            self.mem_block = tf.reshape(self.mem_feat, [split_num, 5, hp['feat_dim']])

            self.mem_node = tf.reduce_mean(self.mem_block, axis=1)

            self.mem_opt = tf.layers.dense(self.mem_node, hp['feat_dim'], activation=tf.nn.relu) + \
                           tf.tile(self.ques_enc, [split_num, 1])

            self.mem_direct = tf.matmul(self.mem_node, self.mem_opt, transpose_b=True) / \
                              (self.mem_node.get_shape().as_list()[-1] ** 0.5)

            self.mem_fw_direct = tf.nn.softmax(self.mem_direct)

            self.mem_bw_direct = tf.nn.softmax(self.mem_direct, axis=0)

            self.mem_ans = tf.reduce_mean(tf.matmul(self.mem_direct, self.mem_node), axis=0, keepdims=True) + \
                           tf.reduce_mean(tf.matmul(self.mem_direct, self.mem_node, transpose_a=True), axis=0,
                                          keepdims=True)

        self.output = tf.reduce_sum(self.mem_ans * self.ans_enc, axis=1)


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
        t, tt = sess.run([model.ques_output[0], model.subt_enc])

        print(t.shape)
        print(tt.shape)


if __name__ == '__main__':
    main()

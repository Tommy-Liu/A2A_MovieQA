import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from config import MovieQAPath
from legacy.input import Input

_mp = MovieQAPath()

hp = {'emb_dim': 512, 'feat_dim': 512,
      'learning_rate': 10 ** (-3), 'decay_rate': 1, 'decay_type': 'inv_sqrt', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1}


def unit_norm(x, dim=2):
    return layers.unit_norm(x, dim=dim, epsilon=1e-12)


def get_cell():
    cell = tf.nn.rnn_cell.GRUCell(hp['emb_dim'])
    # cell = CudnnGRU()
    return cell


class Model(object):
    def __init__(self, data, training=True, **kwargs):
        self.data = data

        with tf.variable_scope('Embedding'):
            self.embedding = tf.get_variable(
                'embedding_matrix', initializer=np.load(_mp.embedding_file), trainable=False)

            self.ques = tf.nn.embedding_lookup(self.embedding, self.data.ques)  # (1, L_q, E)
            self.ans = tf.nn.embedding_lookup(self.embedding, self.data.ans)  # (5, L_a, E)
            self.subt = tf.nn.embedding_lookup(self.embedding, self.data.subt)  # (N, L_s, E)

        with tf.variable_scope('Language_Encode'):
            fw_cell, bw_cell = get_cell(), get_cell()  # (state (E_t), attn(E_t), attn_state (E_t * attn_length))

            self.ques_output, self.ques_states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.ques, self.data.ql, dtype=tf.float32)

            self.ans_output, self.ans_states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.ans, self.data.al, dtype=tf.float32)

            self.subt_output, self.subt_states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.subt, self.data.sl, dtype=tf.float32)

            self.ques_enc = tf.concat([self.ques_states[0], self.ques_states[1]], axis=1)
            self.ans_enc = tf.concat([self.ans_states[0], self.ans_states[1]], axis=1)
            self.subt_enc = tf.concat([self.subt_states[0], self.subt_states[1]], axis=1)

            self.ques_enc = unit_norm(self.ques_enc, dim=1)
            self.ans_enc = unit_norm(self.ans_enc, dim=1)
            self.subt_enc = unit_norm(self.subt_enc, dim=1)

        self.summarize = unit_norm(tf.reduce_mean(self.subt_enc, axis=0, keepdims=True), dim=1)  # (1, 2 * E_t)

        self.ans_vec = unit_norm(self.summarize + self.ques_enc, dim=1)  # (1, 2 * E_t)

        self.output = tf.matmul(self.ans_vec, self.ans_enc, transpose_b=True)  # (1, 5)


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

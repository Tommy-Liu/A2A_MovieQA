import numpy as np
import tensorflow as tf

from config import MovieQAPath
from input import Input

_mp = MovieQAPath()


class Model(object):
    def __init__(self):
        self.data = Input()
        self.embedding = tf.get_variable(
            'embedding', initializer=np.load(_mp.embedding_file), trainable=False)

        self.ques_embedding = tf.layers.dense(get_embedding(self.embedding, self.data.ques), 128)
        self.ans_embedding = tf.layers.dense(get_embedding(self.embedding, self.data.ans), 128, reuse=True)
        self.subt_embedding = tf.layers.dense(get_embedding(self.embedding, self.data.subt), 128, reuse=True)

        self.qs_correlation = tf.tensordot(self.ques_embedding, self.subt_embedding, axes=[[2], [2]])
        
        # with tf.variable_scope('Language_Encode'):
        #     gru_cell = tf.nn.rnn_cell.GRUCell(128, kernel_initializer=tf.orthogonal_initializer())
        #     # gru_cell = tf.nn.rnn_cell.ResidualWrapper(gru_cell)
        #
        #     _, self.ques_enc = tf.nn.dynamic_rnn(cell=gru_cell, inputs=self.ques_embedding,
        #                                          sequence_length=self.data.ql, dtype=tf.float32)
        #
        #     _, self.ans_enc = tf.nn.dynamic_rnn(cell=gru_cell, inputs=self.ans_embedding,
        #                                         sequence_length=self.data.al, dtype=tf.float32)
        #
        #     _, self.subt_enc = tf.nn.dynamic_rnn(cell=gru_cell, inputs=self.subt_embedding,
        #                                          sequence_length=self.data.sl, dtype=tf.float32)
        #
        # self.feat = tf.layers.dense(self.data.feat, 128)
        #
        # with tf.variable_scope('Feature_Attention'):
        #     self.qs_aware = tf.nn.tanh(tf.layers.dense(self.subt_enc, 128, use_bias=False) +
        #                                tf.tile(tf.layers.dense(self.ques_enc, 128, use_bias=False),
        #                                        [tf.shape(self.subt_enc)[0], 1]))
        #     self.feat_attention = tf.expand_dims(
        #         tf.nn.softmax(tf.einsum('ijk,ik->ij', self.feat, self.qs_aware), axis=1), axis=2)
        #     self.feat_weighted_sum = tf.reduce_sum(self.feat * self.feat_attention, axis=1)
        #
        # with tf.variable_scope('Subtitle_Attention'):
        #     self.qf_aware = tf.nn.tanh(tf.layers.dense(self.feat_weighted_sum, 128, use_bias=False) +
        #                                tf.tile(tf.layers.dense(self.ques_enc, 128, use_bias=False),
        #                                        [tf.shape(self.feat_weighted_sum)[0], 1]))
        #     self.subt_attention = tf.expand_dims(tf.nn.softmax(tf.einsum('ij,ij->i')), axis=1)


def get_embedding(embedding, ind):
    return tf.nn.embedding_lookup(embedding, ind)


def main():
    model = Model()

    for v in tf.global_variables():
        print(v)
    with tf.Session() as sess:
        sess.run([model.data.initializer, tf.global_variables_initializer()],
                 feed_dict={model.data.record_placeholder: model.data.train_files})
        #
        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        f = sess.run(model.qs_correlation)
        print(f)
        print(f.shape)


if __name__ == '__main__':
    main()

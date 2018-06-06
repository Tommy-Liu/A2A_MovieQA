import numpy as np
import tensorflow as tf

from config import MovieQAPath
from legacy.input import Input

_mp = MovieQAPath()
hp = {'emb_dim': 512, 'feat_dim': 512,
      'learning_rate': 10 ** (-3), 'decay_rate': 0.83, 'decay_type': 'exp', 'decay_epoch': 2,
      'opt': 'adam', 'checkpoint': '', 'dropout_rate': 0.1, 'pos_len': 35}


def dropout(x, training):
    return tf.layers.dropout(x, hp['dropout_rate'], training=training)


def make_mask(x, length):
    return tf.tile(tf.expand_dims(tf.sequence_mask(x, maxlen=length),
                                  axis=-1), [1, 1, hp['emb_dim']])


def safe_mean(x, length):
    length = tf.reshape(tf.to_float(tf.maximum(tf.constant(1, dtype=tf.int64), length)), [-1, 1, 1])
    return tf.reduce_sum(x, axis=1, keepdims=True) / length


def conv_encode(x, length, scope):
    with tf.variable_scope(scope):
        attn = tf.layers.conv1d(x, filters=hp['emb_dim'] / 2, kernel_size=3, padding='same', activation=tf.nn.relu)
        attn = tf.layers.conv1d(attn, filters=1, kernel_size=3, padding='same', dilation_rate=2, activation=None)
        attn = tf.nn.softmax(attn, axis=1)
    return safe_mean(x * (1 + attn), length)


class Model(object):
    def __init__(self, data, training=False):
        self.data = data
        self.initializer = tf.orthogonal_initializer()
        q_mask = make_mask(self.data.ql, 25)  # (1, L_q, E)
        s_mask = make_mask(self.data.sl, 29)  # (N, L_s, E)
        a_mask = make_mask(self.data.al, 34)  # (5, L_a, E)

        ques_shape = tf.shape(q_mask)
        subt_shape = tf.shape(s_mask)
        ans_shape = tf.shape(a_mask)

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
            self.ques_embedding = self.embedding_linear(self.ques, 'question')  # (1, L_q, E_t)
            self.ans_embedding = self.embedding_linear(self.ans, 'answer')  # (5, L_a, E_t)
            self.subt_embedding = self.embedding_linear(self.subt, 'subtitle')  # (N, L_s, E_t)
            self.feat_embedding = tf.layers.dense(self.data.feat, 1024, activation=tf.nn.tanh)  # (N, 64, 1024)

        with tf.variable_scope('Language_Encode'):
            self.ques_enc = conv_encode(self.ques_embedding, self.data.ql, 'question')  # (1, 1, E_t)
            self.subt_enc = conv_encode(self.subt_embedding, self.data.sl, 'subtitle')  # (N, 1, E_t)
            self.ans_enc = conv_encode(self.ans_embedding, self.data.al, 'answer')  # (5, 1, E_t)

        with tf.variable_scope('Language_Attention'):
            shape = tf.shape(self.subt_embedding)
            q = tf.tile(self.ques_enc, [shape[0], shape[1], 1])  # (N, L_s, E_t)
            q = tf.where(s_mask, q, tf.zeros_like(self.subt_embedding))  # (N, L_s, E_t)
            self.sq_concat = tf.concat([self.subt_embedding, q], axis=-1)  # (N, L_s, 2 * E_t)
            self.lang_attn = tf.layers.conv1d(self.sq_concat, filters=hp['feat_dim'], kernel_size=3,
                                              padding='same', activation=tf.nn.relu)  # (N, L_s, E_t)
            self.lang_attn = tf.layers.conv1d(self.lang_attn, filters=1, kernel_size=5, padding='same',
                                              dilation_rate=2, activation=None)  # (N, L_s, 1)
            self.lang_attn = tf.nn.softmax(self.lang_attn, axis=1)  # (N, L_s, 1)
            self.subt_attn_enc = safe_mean(self.subt_embedding * (1 + self.lang_attn), self.data.sl)  # (N, 1, E_t)

        alpha = tf.layers.dense(self.ques_enc, 1, activation=tf.nn.sigmoid)  # (1, 1, 1)
        self.subt_sum = alpha * self.subt_enc + (1 - alpha) * self.subt_attn_enc  # (N, 1, E_t)

        with tf.variable_scope('Visual_Attention'):
            self.tiled_subt = tf.tile(self.subt_sum, [1, 64, 1])  # (N, 64, E_t)
            self.tiled_ques = tf.tile(self.ques_enc, [shape[0], 64, 1])  # (N, 64, E_t)

            self.fsq_concat = tf.concat([self.feat_embedding,
                                         self.tiled_subt, self.tiled_ques], axis=-1)  # (N, 64, 1024 + 2 * E_t)

            self.vis_attn = tf.layers.conv1d(self.fsq_concat, filters=512, kernel_size=1,
                                             padding='same', activation=tf.nn.relu)  # (N, 64, 512)
            self.vis_attn = tf.layers.conv1d(self.vis_attn, filters=1, kernel_size=1,
                                             padding='same', activation=None)  # (N, 64, 1)
            self.vis_attn = tf.nn.softmax(self.vis_attn, axis=1)  # (N, 64, 1)

            self.vis_feat = tf.reduce_sum(self.feat_embedding * self.vis_attn, axis=1, keepdims=True)  # (N, 1, 1024)

        with tf.variable_scope('Temporal_Attention'):
            self.vs_concat = tf.concat([self.vis_feat, self.subt_sum,
                                        tf.tile(self.ques_enc, [shape[0], 1, 1])], axis=-1)  # (N, 1, 1024 + 2 * E_t)

    def embedding_linear(self, x, scope):
        with tf.variable_scope(scope):
            x = tf.layers.dense(x, hp['emb_dim'], activation=tf.nn.tanh, use_bias=False)
        return x

    def dense_wo_everything(self, x):
        return tf.layers.dense(x, hp['emb_dim'], use_bias=False, kernel_initializer=self.initializer)


def main():
    data = Input(split='train')
    model = Model(data)

    for v in tf.global_variables():
        print(v)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        sess.run([model.data.initializer, tf.global_variables_initializer()], )

        # q, a, s = sess.run([model.ques_enc, model.ans_enc, model.subt_enc])
        # print(q.shape, a.shape, s.shape)
        # a, b, c, d = sess.run(model.tri_word_encodes)
        # print(a, b, c, d)
        # print(a.shape, b.shape, c.shape, d.shape)
        a, b = sess.run([model.vis_feat, model.tiled_ques])
        print(a, b)
        print(a.shape, b.shape)


if __name__ == '__main__':
    main()

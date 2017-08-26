import numpy as np
import tensorflow as tf

from config import ModelConfig

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 4, '')

FLAGS = tf.app.flags.FLAGS

time_steps = 20


class VLLabMemoryModel(object):
    def __init__(self):
        self.config = ModelConfig()

        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self.ques_embeddings = None
        self.ans_embeddings = None
        self.subt_embedding = None

        self.build_seq_embedding()

    def build_seq_embedding(self):
        with tf.variable_scope("ques_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.size_vocab_q, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
            self.ques_embeddings = seq_embeddings

        with tf.variable_scope("ans_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.size_vocab_q, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
            self.ans_embeddings = seq_embeddings
        with tf.variable_scope("subt_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.size_vocab_q, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
            self.subt_embedding = seq_embeddings


def main(_):
    batch_features = tf.convert_to_tensor(np.random.rand(4, time_steps, 1536), dtype=tf.float32)

    mask = tf.convert_to_tensor(np.array([
        [1] * i + [0] * (time_steps - i) for i in range(1, FLAGS.batch_size + 1)
    ]), dtype=tf.int64)

    initializer = tf.random_uniform_initializer(
        minval=-0.08,
        maxval=0.08)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(512, initializer=initializer)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                              input_keep_prob=0.7,
                                              output_keep_prob=0.7)

    zero_state = lstm_cell.zero_state(
        batch_size=FLAGS.batch_size, dtype=tf.float32)

    sequence_length = tf.reduce_sum(mask, 1)
    lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                        inputs=batch_features,
                                        sequence_length=sequence_length,
                                        initial_state=zero_state,
                                        dtype=tf.float32)

    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    print('Start extract !!')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        l_o = sess.run(lstm_outputs)
        print(l_o[:, :4, :].shape)
        print(l_o[:, :4, :])


if __name__ == '__main__':
    tf.app.run()

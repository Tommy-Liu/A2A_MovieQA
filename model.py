
import numpy as np
import tensorflow as tf



flags = tf.app.flags

flags.DEFINE_integer('batch_size', 4, '')

FLAGS = tf.app.flags.FLAGS

def main(_):

    batch_features = [np.random.rand(4, 1536) for _ in range(1, FLAGS.batch_size + 1)]

    mask = np.array([
        [1] * i + [0] * (FLAGS.batch_size-i) for i in range(1, FLAGS.batch_size + 1)
    ])
    lstm_cell = tf.contrib



if __name__ == '__main__':
    tf.app.run()

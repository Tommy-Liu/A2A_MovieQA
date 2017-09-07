from glob import glob
from os.path import join

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "movieqa", "")
flags.DEFINE_string("dataset_dir", "./dataset", "")
FLAGS = flags.FLAGS

_TFRECORD_PATTERN = '*.tfrecord'


def main():
    file_names = glob(join())
    file_name_queue = tf.train.string_input_producer()


if __name__ == '__main__':
    tf.app.run()

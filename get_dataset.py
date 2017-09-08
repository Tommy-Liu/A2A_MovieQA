from glob import glob
from os.path import join

import tensorflow as tf

from data_utils import FILE_PATTERN, qa_feature_parsed

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "movieqa", "")
flags.DEFINE_string("dataset_dir", "./dataset", "")
flags.DEFINE_bool("is_training", True, "")
FLAGS = flags.FLAGS

TFRECORD_PATTERN = FILE_PATTERN.replace('%05d-of-%05d', '*')

if FLAGS.is_training:
    TFRECORD_PATTERN = 'training_' + TFRECORD_PATTERN


class MovieQAData(object):
    def __init__(self):
        self.file_names = glob(join(FLAGS.dataset_dir,
                                    TFRECORD_PATTERN % (FLAGS.dataset_name, 'train')))
        file_name_queue = tf.train.string_input_producer(self.file_names)
        reader = tf.TFRecordReader()
        _, example = reader.read(file_name_queue)
        context_features, sequence_features = qa_feature_parsed()
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        ques = tf.sparse_tensor_to_dense(context_parsed['ques'])


def main(_):
    print(TFRECORD_PATTERN % (FLAGS.dataset_name, 'train'))
    file_names = glob(join(FLAGS.dataset_dir,
                           TFRECORD_PATTERN % (FLAGS.dataset_name, 'train')))

    file_name_queue = tf.train.string_input_producer(file_names)
    reader = tf.TFRecordReader()
    _, example = reader.read(file_name_queue)
    context_features, sequence_features = qa_feature_parsed()
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )


if __name__ == '__main__':
    tf.app.run()

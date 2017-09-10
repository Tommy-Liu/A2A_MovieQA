from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf

from config import TrainingConfig
from data_utils import FILE_PATTERN, qa_feature_parsed

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "movieqa", "")
flags.DEFINE_string("dataset_dir", "./dataset", "")
flags.DEFINE_bool("is_training", True, "")
FLAGS = flags.FLAGS

TFRECORD_PATTERN = FILE_PATTERN.replace('%05d-of-%05d', '*')

if FLAGS.is_training:
    TFRECORD_PATTERN = 'training_' + TFRECORD_PATTERN


def stack_batch(ques, subt, length, feat):
    ques = np.stack([ques, ques])
    subt = np.stack([subt, subt])
    feat = np.stack([feat, feat])
    length = np.stack([length, length])
    label = np.array([1, 0])
    return ques, subt, length, feat, label


class MovieQAData(object):
    def __init__(self, config):
        self.config = config
        self.file_names = glob(join(FLAGS.dataset_dir,
                                    TFRECORD_PATTERN % (FLAGS.dataset_name, 'train')))
        file_name_queue = tf.train.string_input_producer(self.file_names,
                                                         num_epochs=1, )
        # min_after_dequeue = 64
        # capacity = min_after_dequeue + (self.config.num_worker + 4) * self.config.batch_size
        reader = tf.TFRecordReader()
        _, example = reader.read(file_name_queue)
        context_features, sequence_features = qa_feature_parsed()
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        ques = tf.sparse_tensor_to_dense(context_parsed['ques'])
        ans = tf.sparse_tensor_to_dense(sequence_parsed['ans'])
        subt = tf.sparse_tensor_to_dense(sequence_parsed['subt'])
        length = tf.sparse_tensor_to_dense(context_parsed['subt_length'])
        ques, subt, length, feat, label = tf.py_func(stack_batch, [ques, subt, length, sequence_parsed['feat']],
                                                     [tf.int64, tf.int64, tf.int64, tf.float32, tf.int64])
        self.ques, self.ans, self.subt, self.length, self.feat, self.label = \
            ques, ans, subt, length, feat, label

        # tensor_list = tf.train.shuffle_batch([ques, ans, subt, length, feat, label],
        #                                      batch_size=self.config.batch_size,
        #                                      capacity=min_after_dequeue +
        #                                               (self.config.num_worker + 4) * self.config.batch_size,
        #                                      min_after_dequeue=min_after_dequeue,
        #                                      num_threads=self.config.num_worker,
        #                                      enqueue_many=True,
        #                                      allow_smaller_final_batch=True)
        # self.ques, self.ans, self.subt, self.length, self.feat, self.label = tensor_list


def test(_):
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
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            na, pa, q, sl, f, s, cs = sess.run([context_parsed['neg_ans'],
                                                context_parsed['pos_ans'],
                                                context_parsed['ques'],
                                                context_parsed['subt_length'],
                                                sequence_parsed['feat'],
                                                sequence_parsed['subt'],
                                                tf.sparse_tensor_to_dense(sequence_parsed['subt']), ])
            print(na, pa, q, sl, f[:3], s[:3], cs[:3], sep='\n')
        coord.request_stop()
        coord.join(threads)


def main(_):
    train_config = TrainingConfig()
    movieqa_data = MovieQAData(train_config)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            tensor_list = sess.run([movieqa_data.ques,
                                    movieqa_data.ans,
                                    movieqa_data.subt,
                                    movieqa_data.feat,
                                    movieqa_data.label])
            q, a, s, f, l = tensor_list
            print(q, a, s.shape, f.shape, l)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()

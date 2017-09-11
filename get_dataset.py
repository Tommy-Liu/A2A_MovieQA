from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf

from config import MovieQAConfig
from data_utils import FILE_PATTERN, qa_feature_parsed

flags = tf.app.flags
flags.DEFINE_bool("is_training", True, "")
FLAGS = flags.FLAGS


TFRECORD_PATTERN = FILE_PATTERN.replace('%05d-of-%05d', '*')

if FLAGS.is_training:
    TFRECORD_PATTERN = 'training_' + TFRECORD_PATTERN


def stack_batch(ques, ques_length, subt, subt_length, feat):
    ques = np.stack([ques, ques])
    # subt = np.stack([subt, subt])
    feat = np.stack([feat, feat])
    subt_length = np.stack([subt_length, subt_length])
    # ques_length = np.array([ques_length, ques_length], dtype=np.int64)
    label = np.array([1, 0])
    return ques, ques_length, subt, subt_length, feat, label


class MovieQAData(object):
    def __init__(self, is_training=True):
        self.config = MovieQAConfig()
        self.file_names = glob(join(self.config.dataset_dir,
                                    TFRECORD_PATTERN % (self.config.dataset_name, 'train')))
        self.num_samples = 0
        # self.get_sample_num()
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
        if is_training:
            # self.sparse_ans = sequence_parsed['ans']
            ques = tf.sparse_tensor_to_dense(context_parsed['ques'])
            ques = tf.stack([ques, ques])
            ques_length = context_parsed['ques_length']
            ques_length = tf.stack([ques_length, ques_length])
            ans = tf.sparse_tensor_to_dense(sequence_parsed['ans'])
            ans_length = tf.sparse_tensor_to_dense(context_parsed['ans_length'])
            subt = tf.sparse_tensor_to_dense(sequence_parsed['subt'])
            subt_length = tf.sparse_tensor_to_dense(context_parsed['subt_length'])
            feat = sequence_parsed['feat']
            label = tf.constant([1, 0], dtype=tf.int64)
            # ques, ques_length, subt, subt_length, feat, label = \
            #     tf.py_func(stack_batch, [ques, ques_length, subt, subt_length, sequence_parsed['feat']],
            #                [tf.int64, tf.int64, tf.int64, tf.int64, tf.float32, tf.int64])
            self.ques, self.ques_length, self.ans, self.ans_length, \
            self.subt, self.subt_length, self.feat, self.label = \
                ques, ques_length, ans, ans_length, subt, subt_length, feat, label

            # tensor_list = tf.train.shuffle_batch([ques, ans, subt, length, feat, label],
            #                                      batch_size=self.config.batch_size,
            #                                      capacity=min_after_dequeue +
            #                                               (self.config.num_worker + 4) * self.config.batch_size,
            #                                      min_after_dequeue=min_after_dequeue,
            #                                      num_threads=self.config.num_worker,
            #                                      enqueue_many=True,
            #                                      allow_smaller_final_batch=True)
            # self.ques, self.ans, self.subt, self.length, self.feat, self.label = tensor_list
            # self._set_shape()

    def get_sample_num(self):
        for tfrecord_file in self.file_names:
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                self.num_samples += 1

    def _set_shape(self):
        self.ques.set_shape([self.config.batch_size, None])
        self.ans.set_shape([self.config.batch_size, None])


def main(_):
    movieqa_data = MovieQAData()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # for i in range(5):
        tensor_list = sess.run([movieqa_data.ques,
                                movieqa_data.ques_length,
                                movieqa_data.ans,
                                movieqa_data.ans_length,
                                movieqa_data.subt,
                                movieqa_data.subt_length,
                                movieqa_data.feat,
                                movieqa_data.label])
        print(tensor_list)
        coord.request_stop()
        coord.join(threads)


def test(_):
    config_ = MovieQAConfig()
    print(TFRECORD_PATTERN % (config_.dataset_name, 'train'))
    file_names = glob(join(config_.dataset_dir,
                           TFRECORD_PATTERN % (config_.dataset_name, 'train')))

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


if __name__ == '__main__':
    tf.app.run()

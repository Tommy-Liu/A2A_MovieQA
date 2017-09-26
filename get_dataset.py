from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow.contrib.data as data

import data_utils as du
from config import MovieQAConfig

config = MovieQAConfig()


def train_parser(record):
    context_features, sequence_features = du.qa_feature_parsed()
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    ques = tf.stack([context_parsed['ques'],
                     context_parsed['ques']])
    ques_length = tf.stack([context_parsed['ques_length'],
                            context_parsed['ques_length']])
    ans = sequence_parsed['ans']
    ans_length = context_parsed['ans_length']
    subt = sequence_parsed['subt']
    subt_length = tf.sparse_tensor_to_dense(context_parsed['subt_length'])
    feat = sequence_parsed['feat']
    label = tf.convert_to_tensor(np.expand_dims(np.array([1, 0]), 1), dtype=tf.int64)

    return ques, ques_length, ans, ans_length, subt, subt_length, feat, label


def eval_parser(record):
    context_features, sequence_features = du.qa_eval_feature_parsed()
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    ques = tf.stack([context_parsed['ques'] for _ in range(5)])
    ques_length = tf.stack([context_parsed['ques_length'] for _ in range(5)])
    ans = sequence_parsed['ans']
    ans_length = context_parsed['ans_length']
    subt = sequence_parsed['subt']
    subt_length = tf.sparse_tensor_to_dense(context_parsed['subt_length'])
    feat = sequence_parsed['feat']
    label = context_parsed['correct_index']

    return ques, ques_length, ans, ans_length, subt, subt_length, feat, label


def test_parser(record):
    context_features, sequence_features = du.qa_test_feature_parsed()
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    ques = tf.sparse_tensor_to_dense(context_parsed['ques'])
    ques = tf.stack([ques for _ in range(5)])
    ques_length = context_parsed['ques_length']
    ques_length = tf.stack([ques_length for _ in range(5)])
    ans = tf.sparse_tensor_to_dense(sequence_parsed['ans'])
    ans_length = tf.sparse_tensor_to_dense(context_parsed['ans_length'])
    subt = tf.sparse_tensor_to_dense(sequence_parsed['subt'])
    subt_length = tf.sparse_tensor_to_dense(context_parsed['subt_length'])
    feat = sequence_parsed['feat']
    index = context_features['correct_index']

    return ques, ques_length, ans, ans_length, subt, subt_length, feat, index


class MovieQAData(object):
    def __init__(self, split='train',
                 modality='fixed_num',
                 is_training=True, dummy=False):
        self.file_names = glob(du.get_file_pattern(config.dataset_dir,
                                                   config.dataset_name,
                                                   split, modality, config.num_shards,
                                                   is_training))
        if not dummy:
            self.file_names_placeholder = tf.placeholder(tf.string, shape=[None])
            if is_training:
                parser = train_parser
            elif split != 'test':
                parser = eval_parser
            else:
                parser = test_parser
            dataset = data.TFRecordDataset(self.file_names_placeholder)
            dataset = dataset.map(parser)
            dataset = dataset.shuffle(config.size_shuffle_buffer)
            self.iterator = dataset.make_initializable_iterator()
        self.unpack_data(dummy, split)

    def unpack_data(self, dummy, split):
        if not dummy:
            if split != 'test':
                self.ques, self.ques_length, self.ans, self.ans_length, \
                self.subt, self.subt_length, self.feat, self.label = \
                    self.iterator.get_next()
            else:
                self.ques, self.ques_length, self.ans, self.ans_length, \
                self.subt, self.subt_length, self.feat, self.index = \
                    self.iterator.get_next()
        else:
            self.ques, self.ques_length, self.ans, self.ans_length, \
            self.subt, self.subt_length, self.feat, self.label = self.get_dummy()

    def get_dummy(self):
        outputs = [
            tf.convert_to_tensor(
                np.tile(np.random.randint(config.size_vocab, size=(1, 10)), (config.batch_size, 1)),
                dtype=tf.int64),
            tf.convert_to_tensor([10, 10], dtype=tf.int64),
            tf.convert_to_tensor(
                np.tile(np.random.randint(config.size_vocab, size=(1, 10)), (config.batch_size, 1)),
                dtype=tf.int64),
            tf.convert_to_tensor([10, 10], dtype=tf.int64),
            tf.convert_to_tensor(np.tile(np.random.randint(config.size_vocab, size=(1, 10)), (10, 1)),
                                 dtype=tf.int64),
            tf.convert_to_tensor([10 for _ in range(10)], dtype=tf.int64),
            tf.convert_to_tensor(np.random.rand(10, 1536), dtype=tf.float32),
            tf.constant([1, 0], dtype=tf.int64, shape=(2, 1))
        ]
        return outputs


def main(_):
    config_ = MovieQAConfig()
    movieqa_data = MovieQAData(FLAGS.split, FLAGS.modality, is_training=FLAGS.is_training)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(movieqa_data.iterator.initializer, feed_dict={
            movieqa_data.file_names_placeholder: movieqa_data.file_names
        })
        i = 0
        try:
            while True:
                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(coord=coord)
                # for i in range(5):
                tensor_list = sess.run([movieqa_data.ques,
                                        movieqa_data.ques_length,
                                        movieqa_data.ans,
                                        movieqa_data.ans_length,
                                        movieqa_data.subt,
                                        movieqa_data.subt_length,
                                        movieqa_data.feat,
                                        movieqa_data.label])
                i += 1
        except tf.errors.OutOfRangeError:
            print(i)
            print(config_.get_num_example(config_.dataset_name, FLAGS.split, FLAGS.modality, FLAGS.is_training))
            print("Done!")
            # coord.request_stop()
            # coord.join(threads)


def test(_):
    TFRECORD_PATTERN = du.FILE_PATTERN.replace('%05d-of-', '*')
    print(TFRECORD_PATTERN % (config.dataset_name, 'train', config.num_shards))
    file_names = glob(join(config.dataset_dir,
                           TFRECORD_PATTERN % (config.dataset_name, 'train')))

    file_name_queue = tf.train.string_input_producer(file_names)
    reader = tf.TFRecordReader()
    _, example = reader.read(file_name_queue)
    context_features, sequence_features = du.qa_feature_parsed()
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
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
    flags = tf.app.flags
    flags.DEFINE_bool("is_training", True, "")
    flags.DEFINE_string("split", "train", "train, test, val")
    flags.DEFINE_string("modality", "fixed_num",
                        "fixed_num, fixed_interval, shot_major, subtitle_major")
    FLAGS = flags.FLAGS
    tf.app.run()

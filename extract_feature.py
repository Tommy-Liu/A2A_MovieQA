import json
import random


import tensorflow as tf

from glob import glob
from tqdm import tqdm
from data_utils import *
from os.path import join
from video_preprocessing import get_base_name
from inception_preprocessing import preprocess_image
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
metadata = './metadata.json'
video_img = './video_img'
IMAGE_PATTERN_ = '*.jpg'
DIR_PATTERN_ = 'tt*'

def main():
    avail_video_metadata = json.load(open('avail_video_metadata.json', 'r'))
    print(avail_video_metadata.keys())

def test():
    writer = tf.python_io.TFRecordWriter('test.tfrecord')

    # for i in range(50):
    frame_feats = float_feature_list([[random.random() for j in range(10)] for k in range(10)])
    context = tf.train.Features(feature={
        "label": int64_feature(0)
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        "frame_feats": frame_feats
    })
    sequence_example = tf.train.SequenceExample(
          context=context, feature_lists=feature_lists)
    # sequence_example = tf.train.Example(features=context)
    writer.write(sequence_example.SerializeToString())
    writer.close()
    # filename_queue = tf.train.string_input_producer(['test.tfrecord'])
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    context_features = {
        "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frame_feats": tf.FixedLenSequenceFeature([10], dtype=tf.float32)
    }
    e = next(tf.python_io.tf_record_iterator('test.tfrecord'))
    context_parsed, sequence_parsed  = tf.parse_single_sequence_example(
            serialized=e,
            context_features=context_features,
            sequence_features=sequence_features
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        l, exa = sess.run([context_parsed['label'], sequence_parsed['frame_feats']])
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # try:
        #     # print(context_parsed['label'],
        #     #       sequence_parsed['frame_feats'])
        #     l, exa= sess.run([context_parsed['label'], sequence_parsed['frame_feats']])
        # except tf.errors.OutOfRangeError:
        #     print('Done!')
        # finally:
        #     coord.request_stop()
        # coord.join(threads)
        print(l, exa)

if __name__ == '__main__':
    main()
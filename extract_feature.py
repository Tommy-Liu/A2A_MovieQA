import json
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from glob import glob
from tqdm import tqdm
from data_utils import *
from os.path import join
from video_preprocessing import get_base_name, exist_make_dirs
from inception_preprocessing import preprocess_image
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

metadata = './metadata.json'
video_img = './video_img'
tf_record_dir = './tfrecords'
IMAGE_PATTERN_ = '*.jpg'
TFRECORD_PATTERN_ = '%s.tfrecord'
DIR_PATTERN_ = 'tt*'
batch_size = 16 * 3

def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)

def get_tf_record_name(video):
    return join(tf_record_dir, TFRECORD_PATTERN_ % video)


def models(images):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes=1001, is_training=False)
    return end_points['PreLogitsFlatten']



# ['map', 'list', 'info', 'subtitle', 'unavailable']
def main():
    exist_make_dirs(tf_record_dir)
    avail_video_metadata = json.load(open('avail_video_metadata.json', 'r'))
    print('Load json file done !!')
    filenames = []
    capacity = []
    tfrecords = []
    for folder in tqdm(avail_video_metadata['list']):
        tfrecords.append(folder)
        imgs = glob(join(video_img, folder, IMAGE_PATTERN_))
        imgs = sorted(imgs)
        capacity.append(len(imgs))
        filenames.extend(imgs)
    # imgs = sorted(glob(join(video_img, 'tt1454029.sf-006466.ef-010607.video', IMAGE_PATTERN_)))
    # print(imgs)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.WholeFileReader()
    _, raw_image = reader.read(filename_queue)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    # # image = tf.image.resize_image_with_crop_or_pad()
    image = preprocess_image(image, 299, 299, is_training=False)
    # print(image)
    min_after_dequeue = batch_size * 4
    images = tf.train.batch([image],
                            batch_size=batch_size,
                            num_threads=4,
                            capacity=2 * min_after_dequeue,
                            allow_smaller_final_batch=True)
    # print(images)
    feature_tensor = make_parallel(models, 3, images=images)
    # print(end_points['PreLogitsFlatten'])

    print('Pipeline setup done !!')
    saver = tf.train.Saver(slim.get_variables_to_restore())
    config = tf.ConfigProto(allow_soft_placement=True, )
    # log_device_placement=True)
    config.gpu_options.allow_growth = True
    print('Start extract !!')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        video_idx = 0
        tfrecord_writer = tf.python_io.TFRecordWriter(get_tf_record_name(tfrecords[video_idx]))
        features_list = []
        count = 0
        try:
            while not coord.should_stop():
                features = sess.run(feature_tensor)
                count += features.shape[0]
                features_list.append(features)
                if count >= capacity[video_idx]:
                    count = count - capacity[video_idx]
                    bound = batch_size - count
                    # ? * 1536
                    final_features = np.concatenate(features_list[:-1] + [features_list[-1][:bound]], 0)
                    print(tfrecords[video_idx], final_features.shape)
                    tfrecord_writer.write(frame_feature_example(final_features).SerializeToString())
                    tfrecord_writer.close()
                    features_list = [features_list[-1][bound:]]
                    video_idx += 1
                    tfrecord_writer = tf.python_io.TFRecordWriter(get_tf_record_name(tfrecords[video_idx]))
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
            coord.join(threads)
            # for img in imgs:
            #     image_data = tf.gfile.FastGFile(img, 'rb').read()


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
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
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

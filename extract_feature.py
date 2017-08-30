import os
import sys
import time
import math
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
from multiprocessing import Manager, Process, Event, Queue

flags = tf.app.flags

flags.DEFINE_string('tf_record_dir', './tfrecords', '')
flags.DEFINE_string('feature_dir', './features', '')
flags.DEFINE_string('metadata', './avail_video_metadata.json', '')
flags.DEFINE_string('video_img', './video_img', '')
flags.DEFINE_integer('num_gpus', 1, '')
flags.DEFINE_integer('per_batch_size', 64, '')
flags.DEFINE_integer('num_worker', 8, '')

FLAGS = flags.FLAGS

filename_json = './filenames.json'
IMAGE_PATTERN_ = '*.jpg'
TFRECORD_PATTERN_ = '%s.tfrecord'
NPY_PATTERN_ = '%s.npy'
DIR_PATTERN_ = 'tt*'
batch_size = FLAGS.per_batch_size * FLAGS.num_gpus
num_worker = FLAGS.num_worker * FLAGS.num_gpus


def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                out_split.append(fn(**{k: v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)


def get_tf_record_name(video):
    return join(FLAGS.tf_record_dir, TFRECORD_PATTERN_ % video)


def get_npy_name(video):
    return join(FLAGS.feature_dir, NPY_PATTERN_ % video)


def models(images):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes=1001, is_training=False)
    return end_points['PreLogitsFlatten']


def get_images_path():
    if not os.path.exists(filename_json):
        avail_video_metadata = json.load(open(FLAGS.metadata, 'r'))
        print('Load json file done !!')
        file_names = []
        file_names_sep = []
        capacity = []
        npy_names = []
        for folder in tqdm(avail_video_metadata['list']):
            # if not os.path.exists(get_npy_name(folder)):
            npy_names.append(get_npy_name(folder))
            imgs = glob(join(FLAGS.video_img, folder, IMAGE_PATTERN_))
            imgs = sorted(imgs)
            capacity.append(len(imgs))
            file_names_sep.append(imgs)
            file_names.extend(imgs)
        json.dump({
            'file_names': file_names_sep,
            'capacity': capacity,
            'npy_names': npy_names,
        }, open(filename_json, 'w'))
    file_names_json = json.load(open(filename_json, 'r'))
    print('Load json file done !!')
    file_names, capacity, npy_names = [], [], []
    for idx, name in enumerate(tqdm(file_names_json['npy_names'])):
        if not os.path.exists(name):
            file_names.extend(file_names_json['file_names'][idx])
            capacity.append(len(file_names_json['file_names'][idx]))
            npy_names.append(name)
    print(len(file_names), len(capacity))
    return file_names, capacity, npy_names


def input_pipeline(filenames):
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=1)  # , capacity=   batch_size * 2)
    reader = tf.WholeFileReader()
    _, raw_image = reader.read(filename_queue)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = preprocess_image(image, 299, 299, is_training=False)
    images = tf.train.batch([image],
                            batch_size=batch_size,
                            num_threads=num_worker,
                            capacity=2 * batch_size * FLAGS.num_worker,
                            allow_smaller_final_batch=True)
    return images


def parse_function(filename):
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = preprocess_image(image, 299, 299, is_training=False)
    return image


def count_num(features_list):
    num = 0
    for featues in features_list:
        num += featues.shape[0]
    return num


def writer_worker(e, features_list, capacity, npy_names):
    total_count = 0
    video_idx = 0
    front_cut = 0
    local_feature_list = []
    while True:
        if len(features_list) > 0:
            local_feature_list.append(features_list.pop(0))
            total_count += local_feature_list[-1].shape[0]
            if total_count >= capacity[video_idx]:
                end_cut = (capacity[video_idx] + front_cut) % batch_size
                list_bound = int(math.floor((capacity[video_idx] + front_cut) / batch_size))
                concat = [local_feature_list[0][front_cut:]] + \
                         local_feature_list[1:list_bound] + \
                         [local_feature_list[list_bound][:end_cut]] if end_cut > 0 else \
                    [local_feature_list[0][front_cut:]] + \
                    local_feature_list[1:list_bound]
                final_features = np.concatenate(concat, 0)
                print(npy_names[video_idx], final_features.shape, capacity[video_idx], len(local_feature_list))
                np.save(npy_names[video_idx], final_features)
                for i in range(list_bound):
                    local_feature_list.pop(0)
                front_cut = end_cut
                total_count -= capacity[video_idx]
                video_idx += 1
        else:
            time.sleep(0.5)
        if len(features_list) == 0:
            e.set()
        else:
            e.clear()  # ['map', 'list', 'info', 'subtitle', 'unavailable']


def main(_):
    exist_make_dirs(FLAGS.feature_dir)
    filenames, capacity, npy_names = get_images_path()
    images = input_pipeline(filenames)
    # file_placeholder = tf.placeholder(tf.string, shape=[None])
    # dataset = tf.contrib.data.Dataset.from_tensor_slices(file_placeholder)
    # dataset = dataset.map(parse_function)
    # dataset = dataset.repeat(1)
    # dataset = dataset.batch(batch_size)
    # iterator = dataset.make_initializable_iterator()
    # images = iterator.get_next()
    # feature_tensor = make_parallel(models, FLAGS.num_gpus, images=images)
    feature_tensor = models(images)
    print('Pipeline setup done !!')
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto()  # allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    print('Start extract !!')
    with tf.Session(config=config) as sess, Manager() as manager:
        e = Event()
        features_list = manager.list()
        p = Process(target=writer_worker, args=(e, features_list, capacity, npy_names))
        p.start()
        # sess.run(iterator.initializer, feed_dict={
        #     file_placeholder: filenames
        # })
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            # while not coord.should_stop():
            while True:
                features_list.append(sess.run(feature_tensor))
        except tf.errors.OutOfRangeError:
            print('done!')
            e.wait()
            time.sleep(3)
            p.terminate()
        except KeyboardInterrupt:
            print()
            p.terminate()
        finally:
            coord.request_stop()
            coord.join(threads)


def test_time():
    exist_make_dirs(FLAGS.tf_record_dir)
    filenames, capacity, npy_names = get_images_path()
    images = input_pipeline(filenames)
    print('Pipeline setup done !!')
    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    print('Start extract !!')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(images)
        avg_time = 0
        iter = 1
        try:
            while not coord.should_stop():
                start_time = time.time()
                sess.run(images)
                end_time = time.time()
                avg_time = avg_time + (end_time - start_time - avg_time) / iter
                sys.stdout.write("\rAverage time: %.4f Iter time: %.4f" % (avg_time, end_time - start_time))
                sys.stdout.flush()
                iter += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        except KeyboardInterrupt:
            print()
        finally:
            coord.request_stop()
            coord.join(threads)


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
    tf.app.run()

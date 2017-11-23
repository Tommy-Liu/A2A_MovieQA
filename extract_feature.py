import random
import sys
import time
from math import ceil
from multiprocessing import Manager, Event, Process
from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

import data_utils as du
from config import MovieQAConfig
from inception_preprocessing import preprocess_image
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

config = MovieQAConfig()

filename_json = config.images_name_file
IMAGE_PATTERN_ = '*.jpg'
IMAGE_FILE_NAME_ = 'img_%05d.jpg'
DIR_PATTERN_ = 'tt*'


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


def models(images):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes=1001, is_training=False)
    return end_points['PreLogitsFlatten']


def get_images_path():
    file_names, capacity, npy_names = [], [], []
    video_data = du.load_json(config.video_data_file)
    for key in tqdm(video_data.keys()):
        if video_data[key]['avail']:
            npy_names.append(du.get_npy_name(config.feature_dir, key))
            imgs = [join(config.video_img_dir, key, IMAGE_FILE_NAME_ % (i + 1))
                    for i in range(video_data[key]['num_frames'])]
            capacity.append(len(imgs))
            file_names.extend(imgs)
    return file_names, capacity, npy_names


def count_num(features_list):
    num = 0
    for featues in features_list:
        num += featues.shape[0]
    return num


def writer_worker(e, features_list, filename_list, capacity, npy_names):
    video_idx = 0
    local_feature = np.zeros((0, config.feature_dim), dtype=np.float32)
    local_filename = []
    # video_subt = json.load(open(config.subtitle_file))
    while True:
        if len(features_list) > 0:
            local_feature = np.concatenate([local_feature, features_list.pop(0)])
            local_filename.extend(filename_list.pop(0))
            if local_feature.shape[0] >= capacity[video_idx]:
                final_features = local_feature[:capacity[video_idx]]
                final_filename = local_filename[:capacity[video_idx]]
                assert final_features.shape[0] == capacity[video_idx], \
                    "%s Both frames are not same!" % npy_names[video_idx]
                assert all([du.get_base_name_without_ext(npy_names[video_idx])
                            == du.get_base_name_without_ext(final_filename[i])
                            for i in range(len(final_features))]), \
                    "Wrong images! %s\n%s" % (npy_names[video_idx], final_filename)
                print(npy_names[video_idx], final_features.shape, capacity[video_idx], len(local_feature))
                # np.save(npy_names[video_idx], final_features)
                local_feature = local_feature[capacity[video_idx]:]
                video_idx += 1
        else:
            time.sleep(0.5)
        if len(features_list) == 0 and video_idx == len(capacity):
            e.set()
        else:
            e.clear()  # ['map', 'list', 'info', 'subtitle', 'unavailable']


def parse_func(filename):
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = preprocess_image(image, 299, 299, is_training=False)
    return image, filename


def input_pipeline(filename_placeholder):
    dataset = tf.data.Dataset.from_tensor_slices(filename_placeholder)
    dataset = dataset.map(parse_func, num_parallel_calls=num_worker)
    dataset = dataset.prefetch(10000)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    images, names = iterator.get_next()
    return images, names, iterator


def main(_):
    du.exist_make_dirs(config.feature_dir)
    filenames, capacity, npy_names = get_images_path()
    num_step = int(ceil(len(filenames) / batch_size))
    filename_placeholder = tf.placeholder(tf.string, shape=[None])
    images, names, iterator = input_pipeline(filename_placeholder)
    feature_tensor = models(images)
    print('Pipeline setup done !!')
    saver = tf.train.Saver(tf.global_variables())
    print('Start extract !!')
    with tf.Session() as sess, Manager() as manager:
        e = Event()
        features_list = manager.list()
        filename_list = manager.list()
        p = Process(target=writer_worker, args=(e, features_list, filename_list, capacity, npy_names))
        p.start()
        sess.run(iterator.initializer, feed_dict={filename_placeholder: filenames})
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')
        try:
            for _ in range(num_step):
                f, n = sess.run([feature_tensor, names])
                # print(n)
                features_list.append(f)
                filename_list.append([str(i) for i in n])
            e.wait()
            time.sleep(3)
            p.terminate()
        except KeyboardInterrupt:
            print()
            p.terminate()
        finally:
            time.sleep(1)
            p.join()


def test_time():
    du.exist_make_dirs(config.feature_dir)
    filenames, capacity, npy_names = get_images_path()
    images = input_pipeline(filenames)
    print('Pipeline setup done !!')
    config_ = tf.ConfigProto(allow_soft_placement=True, )
    config_.gpu_options.allow_growth = True
    print('Start extract !!')
    with tf.Session(config=config_) as sess:
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
    frame_feats = du.float_feature_list([[random.random() for j in range(10)] for k in range(10)])
    context = tf.train.Features(feature={
        "label": du.int64_feature(0)
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

    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.allow_growth = True
    with tf.Session(config=config_) as sess:
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
    flags = tf.app.flags
    flags.DEFINE_integer('num_gpus', 1, '')
    flags.DEFINE_integer('per_batch_size', 64, '')
    flags.DEFINE_integer('num_worker', 8, '')
    flags.DEFINE_string('split', 'train', '')
    FLAGS = flags.FLAGS
    batch_size = FLAGS.per_batch_size * FLAGS.num_gpus
    num_worker = FLAGS.num_worker * FLAGS.num_gpus
    tf.app.run()

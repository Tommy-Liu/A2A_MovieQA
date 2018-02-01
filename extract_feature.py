import argparse
import time
from math import ceil
from multiprocessing import Manager, Event, Process
from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

from config import MovieQAPath
from inception_preprocessing import preprocess_image
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from utils import data_utils as du
from utils import func_utils as fu

mp = MovieQAPath()


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
    return end_points['Conv2d_7b_1x1']


def get_images_path():
    file_names, capacity, npy_names = [], [], []
    video_data = du.json_load(mp.video_data_file)
    for key in tqdm(video_data.keys(), desc='Collect images'):
        npy_names.append(join(mp.feature_dir, key + '.npy'))
        imgs = [join(mp.image_dir, key, '%s_%05d.jpg' % (key, i + 1))
                for i in range(video_data[key]['real_frames'])]
        capacity.append(len(imgs))
        file_names.extend(imgs)
    return file_names, capacity, npy_names


def count_num(features_list):
    num = 0
    for features in features_list:
        num += features.shape[0]
    return num


def writer_worker(e, features_list, capacity, npy_names):
    video_idx = 0
    local_feature = np.zeros((0, 8, 8, 1536), dtype=np.float32)
    local_filename = []
    with tqdm(total=len(npy_names)) as pbar:
        while True:
            if len(features_list) > 0:
                f, n = features_list.pop(0)
                local_feature = np.concatenate([local_feature, f])
                local_filename.extend(n)
                if local_feature.shape[0] >= capacity[video_idx]:
                    final_features = local_feature[:capacity[video_idx]]
                    final_filename = local_filename[:capacity[video_idx]]
                    assert final_features.shape[0] == capacity[video_idx], \
                        "%s Both frames are not same!" % npy_names[video_idx]
                    for i in range(len(final_features)):
                        assert fu.basename_wo_ext(npy_names[video_idx]) == final_filename[i].split('/')[-2], \
                            "Wrong images! %s\n%s" % (npy_names[video_idx], final_filename[i])
                    pbar.set_description(' '.join([fu.basename_wo_ext(npy_names[video_idx]),
                                                   str(final_features.shape),
                                                   str(capacity[video_idx]),
                                                   str(len(local_feature))]))
                    np.save(npy_names[video_idx], final_features)
                    local_feature = local_feature[capacity[video_idx]:]
                    local_filename = local_filename[capacity[video_idx]:]
                    video_idx += 1
                    pbar.update()
            else:
                time.sleep(3)
            if len(features_list) == 0 and video_idx == len(capacity):
                e.set()
            else:
                e.clear()  # ['map', 'list', 'info', 'subtitle', 'unavailable']


def parse_func(filename):
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = preprocess_image(image, 299, 299, is_training=False)
    return image, filename


def input_pipeline(filename_placeholder, batch_size=32, num_worker=4):
    dataset = tf.data.Dataset.from_tensor_slices(filename_placeholder)
    dataset = dataset.map(parse_func, num_parallel_calls=num_worker)
    dataset = dataset.prefetch(10000)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    images, names = iterator.get_next()
    return images, names, iterator


def extract(batch_size, num_worker):
    fu.make_dirs(mp.feature_dir)
    filenames, capacity, npy_names = get_images_path()

    num_step = int(ceil(len(filenames) / batch_size))
    filename_placeholder = tf.placeholder(tf.string, shape=[None])
    images, names, iterator = input_pipeline(filename_placeholder, batch_size, num_worker)
    feature_tensor = models(images)

    print('Pipeline setup done !!')

    saver = tf.train.Saver(tf.global_variables())

    print('Start extract !!')

    with tf.Session() as sess, Manager() as manager:
        e = Event()
        features_list = manager.list()
        p = Process(target=writer_worker, args=(e, features_list, capacity, npy_names))
        p.start()
        sess.run(iterator.initializer, feed_dict={filename_placeholder: filenames})
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')
        try:
            for _ in range(num_step):
                f, n = sess.run([feature_tensor, names])
                # print(n)
                features_list.append((f, [i.decode() for i in n]))
            e.wait()
            time.sleep(3)
            p.terminate()
        except KeyboardInterrupt:
            print()
            p.terminate()
        finally:
            time.sleep(1)
            p.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', default=1)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--num_worker', default=8)
    args = parser.parse_args()
    batch_size = args.batch_size * args.num_gpu
    num_worker = args.num_worker * args.num_gpu
    extract(batch_size, num_worker)


if __name__ == '__main__':
    main()

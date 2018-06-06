import argparse
import os
from functools import partial
from math import ceil
from multiprocessing import Pool
from os.path import join, exists

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from numpy.lib.format import read_array_header_1_0, read_magic
from tqdm import tqdm

from config import MovieQAPath
from legacy.inception_preprocessing import preprocess_image
from legacy.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from utils import data_utils as du
from utils import func_utils as fu

_mp = MovieQAPath()


def make_parallel(model, num_gpus, imgs):
    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                out_split.append(model(images=imgs[i]))
    # return tf.concat(out_split, axis=0)
    return out_split


def models(images):
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes=1001, is_training=False)

    # return tf.nn.max_pool(end_points['Conv2d_7b_1x1'], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    return end_points['Conv2d_7b_1x1']


def load_shape(n):
    with open(n, 'rb') as f:
        major, minor = read_magic(f)
        shape, fortran, dtype = read_array_header_1_0(f)
    if len(shape) != 4:
        raise TypeError('Errr! Single image... %s' % n)
    return shape


def collect(video_data, k):
    imgs = [join(_mp.image_dir, k, '%s_%05d.jpg' % (k, i + 1))
            for i in range(0, video_data[k]['real_frames'], 10)]
    npy_name = join(_mp.feature_dir, k + '.npy')
    if not exists(npy_name) or load_shape(npy_name)[0] != len(imgs):
        return npy_name, len(imgs), imgs
    else:
        return None


def get_images_path():
    file_names, capacity, npy_names = [], [], []
    video_data = dict(item for v in du.json_load(_mp.video_data_file).values() for item in v.items())

    func = partial(collect, video_data)
    with Pool(16) as pool, tqdm(total=len(video_data), desc='Collect images') as pbar:
        for ins in pool.imap_unordered(func, list(video_data.keys())):
            if ins:
                npy_names.append(ins[0])
                capacity.append(ins[1])
                file_names.extend(ins[2])
            pbar.update()
    return file_names, capacity, npy_names


def get_images_path_v2():
    file_names, capacity, npy_names = [], [], []
    video_data = du.json_load(_mp.video_data_file)

    for imdb_key in tqdm(video_data, desc='Collect Images'):
        npy_names.append(join(_mp.feature_dir, imdb_key + '.npy'))
        videos = list(video_data[imdb_key].keys())
        videos.sort()
        num = 0
        for v in tqdm(videos):
            images = [join(_mp.image_dir, v, '%s_%05d.jpg' % (v, i + 1))
                      for i in range(0, video_data[imdb_key][v]['real_frames'], 15)]
            file_names.extend(images)
            num += len(images)
        capacity.append(num)

    print(capacity, npy_names)
    return file_names, capacity, npy_names


def get_images_path_v3():
    file_names, capacity, npy_names = [], [], []
    sample = du.json_load(_mp.sample_frame_file)

    for imdb_key in tqdm(sample, desc='Collect Images'):
        # if not exists(join(_mp.feature_dir, imdb_key + '.npy')):
        npy_names.append(join(_mp.feature_dir, imdb_key + '.npy'))
        videos = list(sample[imdb_key].keys())
        videos.sort()
        num = 0
        for v in tqdm(videos):
            images = [join(_mp.image_dir, v, '%s_%05d.jpg' % (v, i + 1))
                      for i in sample[imdb_key][v]]
            file_names.extend(images)
            num += len(images)
        capacity.append(num)
    # print(capacity, npy_names)
    return file_names, capacity, npy_names


def count_num(features_list):
    num = 0
    for features in features_list:
        num += features.shape[0]
    return num


def writer_worker(queue, capacity, npy_names):
    video_idx = 0
    local_feature = []
    local_filename = []
    with tqdm(total=len(npy_names)) as pbar:
        while len(capacity) > video_idx:
            item = queue.get()
            if item:
                f, n = item
                local_feature.append(f)
                local_filename.extend(n)
                local_size = len(local_filename)
                while len(capacity) > video_idx and local_size >= capacity[video_idx]:

                    concat_feature = np.concatenate(local_feature, axis=0)
                    final_features = concat_feature[:capacity[video_idx]]
                    final_filename = local_filename[:capacity[video_idx]]

                    assert final_features.shape[0] == capacity[video_idx], \
                        "%s Both frames are not same!" % npy_names[video_idx]
                    for i in range(len(final_features)):
                        assert fu.basename_wo_ext(npy_names[video_idx]) == \
                               fu.basename_wo_ext(final_filename[i]).split('.')[0], \
                            "Wrong images! %s\n%s" % (npy_names[video_idx], final_filename[i])
                    try:
                        np.save(npy_names[video_idx], final_features)
                    except Exception as e:
                        np.save(npy_names[video_idx], final_features)
                        raise e
                    pbar.set_description(' '.join([fu.basename_wo_ext(npy_names[video_idx]),
                                                   str(len(final_features))]))
                    del local_feature[:]
                    local_feature.append(concat_feature[capacity[video_idx]:])
                    local_filename = local_filename[capacity[video_idx]:]
                    local_size = len(local_filename)
                    video_idx += 1
                    pbar.update()
            else:
                break


def parse_func(filename):
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    return image, filename


def preprocess_func(image, filename):
    image = preprocess_image(image, 299, 299, is_training=False)
    return image, filename


def input_pipeline(filename_placeholder, batch_size=32, num_worker=4):
    dataset = tf.data.Dataset.from_tensor_slices(filename_placeholder)
    dataset = dataset.repeat(1)
    images, names_, iterator = [], [], []
    if args.num_gpu > 1:
        for i in range(args.num_gpu):
            dd = dataset.shard(args.num_gpu, i)
            dd = dd.map(parse_func, num_parallel_calls=num_worker)
            dd = dd.map(preprocess_func, num_parallel_calls=num_worker)
            dd = dd.batch(batch_size)
            dd = dd.prefetch(16)
            itit = dd.make_initializable_iterator()
            im, n_ = itit.get_next()
            images.append(im)
            names_.append(n_)
            iterator.append(itit)
    else:
        dataset = dataset.map(parse_func, num_parallel_calls=num_worker)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1000)
        iterator = dataset.make_initializable_iterator()
        images, names_ = iterator.get_next()
    return images, names_, iterator


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', default=1, type=int, help='Number of GPU going to be used')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size of images')
    parser.add_argument('--num_worker', default=2, type=int, help='Number of worker reading data.')
    parser.add_argument('--reset', action='store_true', help='Reset all the extracted features.')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    bsize = args.batch_size
    num_w = args.num_worker
    reset = args.reset
    if reset:
        os.system('rm -rf %s' % _mp.feature_dir)
    fu.make_dirs(_mp.feature_dir)
    fn, cap, npy_n = get_images_path_v3()

    num_step = int(ceil(len(fn) / bsize))
    fp = tf.placeholder(tf.string, shape=[None])
    img, names, it = input_pipeline(fp, bsize, num_w)
    if args.num_gpu > 1:
        feature_tensor = make_parallel(models, args.num_gpu, img)
    else:
        feature_tensor = models(img)

    print('Pipeline setup done !!')

    saver = tf.train.Saver(tf.global_variables())

    print('Start extract !!')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        # q = Queue()
        # p = Process(target=writer_worker, args=(q, cap, npy_n))
        # p.start()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')
        sess.run([iit.initializer for iit in it], feed_dict={fp: fn})
        # print(sess.run(feature_tensor).shape)
        try:
            for _ in range(num_step):
                feature, name = sess.run([feature_tensor, names])
                # q.put((feature, [i.decode() for i in name]))
            # q.put(None)
        except KeyboardInterrupt:
            print()
            # p.terminate()
        finally:
            pass
            # p.join()
            # q.close()

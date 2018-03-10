import argparse
from functools import partial
from math import ceil
from multiprocessing import Pool, Queue, Process
from os.path import join, exists

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from numpy.lib.format import read_array_header_1_0, read_magic
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


def load_shape(name):
    with open(name, 'rb') as f:
        major, minor = read_magic(f)
        shape, fortran, dtype = read_array_header_1_0(f)
    if len(shape) != 4:
        raise TypeError('Errr! Single image... %s' % name)
    return shape


def collect(video_data, k):
    imgs = [join(mp.image_dir, k, '%s_%05d.jpg' % (k, i + 1))
            for i in range(0, video_data[k]['real_frames'], 10)]
    npy_name = join(mp.feature_dir, k + '.npy')
    if not exists(npy_name) or load_shape(npy_name)[0] != len(imgs):
        return npy_name, len(imgs), imgs
    else:
        return None


def get_images_path():
    file_names, capacity, npy_names = [], [], []
    video_data = dict(item for v in du.json_load(mp.video_data_file).values() for item in v.items())

    func = partial(collect, video_data)
    with Pool(16) as p, tqdm(total=len(video_data), desc='Collect images') as pbar:
        for ins in p.imap_unordered(func, list(video_data.keys())):
            if ins:
                npy_names.append(ins[0])
                capacity.append(ins[1])
                file_names.extend(ins[2])
            pbar.update()
    return file_names, capacity, npy_names


def count_num(features_list):
    num = 0
    for features in features_list:
        num += features.shape[0]
    return num


def writer_worker(queue, capacity, npy_names):
    video_idx = 0
    local_feature = np.zeros((0, 8, 8, 1536), dtype=np.float32)
    local_filename = []
    with tqdm(total=len(npy_names)) as pbar:
        while True:
            item = queue.get()
            if item:
                f, n = item
                local_feature = np.concatenate([local_feature, f])
                local_filename.extend(n)
                while len(capacity) > video_idx and \
                        local_feature.shape[0] >= capacity[video_idx]:

                    final_features = local_feature[range(capacity[video_idx])]
                    final_filename = local_filename[:capacity[video_idx]]

                    assert final_features.shape[0] == capacity[video_idx], \
                        "%s Both frames are not same!" % npy_names[video_idx]
                    for i in range(len(final_features)):
                        assert fu.basename_wo_ext(npy_names[video_idx]) == final_filename[i].split('/')[-2], \
                            "Wrong images! %s\n%s" % (npy_names[video_idx], final_filename[i])
                    norm = np.linalg.norm(final_features, axis=3, keepdims=True)
                    norm = np.select([norm > 0], [norm], default=1.)
                    final_features = final_features / norm
                    try:
                        np.save(npy_names[video_idx], final_features)
                    except Exception as e:
                        np.save(npy_names[video_idx], final_features)
                        raise e
                    pbar.set_description(' '.join([fu.basename_wo_ext(npy_names[video_idx])[:20],
                                                   str(len(final_features))]))
                    local_feature = local_feature[range(capacity[video_idx], len(local_feature))]
                    local_filename = local_filename[capacity[video_idx]:]
                    video_idx += 1
                    pbar.update()
            else:
                break


def parse_func(filename):
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    image = preprocess_image(image, 299, 299, is_training=False)
    return image, filename


def input_pipeline(filename_placeholder, batch_size=32, num_worker=4):
    dataset = tf.data.Dataset.from_tensor_slices(filename_placeholder)
    dataset = dataset.repeat(1)
    dataset = dataset.map(parse_func, num_parallel_calls=num_worker)
    dataset = dataset.prefetch(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(100)
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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        queue = Queue()
        p = Process(target=writer_worker, args=(queue, capacity, npy_names))
        p.start()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')
        sess.run(iterator.initializer, feed_dict={filename_placeholder: filenames})
        try:
            for _ in range(num_step):
                f, n = sess.run([feature_tensor, names])
                queue.put((f, [i.decode() for i in n]))
            queue.put(None)
        except KeyboardInterrupt:
            print()
            p.terminate()
        finally:
            p.join()
            queue.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    args = parser.parse_args()
    batch_size = args.batch_size * args.num_gpu
    num_worker = args.num_worker * args.num_gpu
    extract(batch_size, num_worker)


if __name__ == '__main__':
    main()

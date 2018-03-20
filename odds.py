from glob import glob
from multiprocessing import Queue, Process
from os.path import join

import numpy as np

from config import MovieQAPath
from extract_feature import load_shape
from utils import data_utils as du

_mp = MovieQAPath()


class LUL:
    pass


def consumer(queue):
    while True:
        n = queue.get()
        if n:
            print(n)
        else:
            break


def multiprocess():
    q = Queue()
    p = Process(target=consumer, args=(q,))
    p.start()

    for i in range(1, 20):
        q.put(i)
    q.put(None)

    p.join()
    q.close()


def normalize_save(file):
    feat = np.load(file)
    norm = np.linalg.norm(feat, axis=3, keepdims=True)
    norm = np.select([norm > 0], [norm], default=1.)

    norm_feat = feat / norm

    np.save(file, norm_feat)


def main():
    features = glob(join(_mp.feature_dir, '*.npy'))
    video_data = du.json_load(_mp.video_data_file)

    imdb_feat = []

    for imdb_key in video_data:
        imdb_feat.append((imdb_key, [f for f in features if imdb_key in f]))

    for feat in imdb_feat:
        length = 0
        for f in feat[1]:
            length += load_shape(f)[0]
        print(feat[0], '%.3f' % (length * 4 * 4 * 1536 / 1024 / 1024 / 1024), 'GB')

    # with Pool(8) as pool, tqdm(total=len(features), desc='Normalize features') as pbar:
    #     for _ in pool.imap_unordered(normalize_save, features):
    #         pbar.update()

    # feat = np.load(features[0])
    # norm = np.linalg.norm(feat, axis=3, keepdims=True)
    # print(norm.shape)
    # norm = np.select([norm > 0], [norm], default=1.0)
    # print(norm.shape)
    # print(feat.shape)
    #
    # norm_feat = feat / norm
    #
    # print(np.linalg.norm(norm_feat, axis=3).shape)
    # print(norm_feat.shape)
    # print(feat.shape)
    # print(norm_feat.shape == feat.shape)


if __name__ == '__main__':
    main()

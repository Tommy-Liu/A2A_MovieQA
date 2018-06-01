from multiprocessing import Queue, Process

import numpy as np

from config import MovieQAPath
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
    sample = du.json_load(_mp.sample_frame_file)

    for k in sample:
        print(k, '%.3f' % (sum([len(sample[k][v]) for v in sample[k]]) * 4 * 4 * 1536 / 1024 / 1024 / 1024), "Gb")
    # features = glob(join(_mp.feature_dir, '*.npy'))
    # video_data = du.json_load(_mp.video_data_file)
    #
    # imdb_feat = []
    #
    # for imdb_key in video_data:
    #     imdb_feat.append((imdb_key, [f for f in features if imdb_key in f]))
    #
    # for feat in imdb_feat:
    #     length = 0
    #     for f in feat[1]:
    #         length += load_shape(f)[0]
    #     print(feat[0], '%.3f' % (length * 4 * 4 * 1536 / 1024 / 1024 / 1024), 'GB')

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

# import argparse
# import os
# import matplotlib
# import textwrap
# import numpy as np
# from tqdm import trange
# np.set_printoptions(threshold=np.inf)
# matplotlib.use('Agg')
#
# from multiprocessing import Pool
# import matplotlib.pyplot as plt
#
# import seaborn as sns
#
# sns.set()
#
# # import utils.func_utils as fu
# import utils.data_utils as du
#
# from config import MovieQAPath
# from data.data_loader import QA, Subtitle
#
# _mp = MovieQAPath()
#
# def train_save(ins):
#     attn = train_attn[ins['qid'].replace(':', '')]
#     pair = train_pair[ins['qid'].replace(':', '')][:]
#     # print(attn.shape)
#     # print(pair.shape)
#     spec = np.load(os.path.join(_mp.encode_dir, ins['qid'] + '_spec' + '.npy'))
#     # print(spec)
#     iid = [idx for i, idx in enumerate(index[ins['imdb_key']]) if spec[i] == 1]
#     # print(iid)
#     sentences = [textwrap.fill(subtitle[ins['imdb_key']]['lines'][idx], 40) for idx in iid]
#     # print(sentences)
#
#     plt.figure(figsize=(6, int(attn.shape[0] / 2)))
#     ax = sns.heatmap(attn, vmin=0, vmax=1)
#     ax.set_xticklabels([textwrap.fill(ins['question'], 50)] +
#                        [textwrap.fill(ans, 50) for ans in ins['answers']], rotation=45, ha='right')
#     ax.set_yticklabels(sentences, rotation=0)
#     ax.set_title('%d %d' % (pair[0], pair[1]))
#     plt.savefig('fig/%s.jpg' % ins['qid'].replace(':', ''), bbox_inches='tight')
#     plt.close()
#
# def val_save(ins):
#     attn = val_attn[ins['qid'].replace(':', '')]
#     pair = val_pair[ins['qid'].replace(':', '')][:]
#     # print(attn.shape)
#     # print(pair.shape)
#     spec = np.load(os.path.join(_mp.encode_dir, ins['qid'] + '_spec' + '.npy'))
#     # print(spec)
#     iid = [idx for i, idx in enumerate(index[ins['imdb_key']]) if spec[i] == 1]
#     # print(iid)
#     sentences = [textwrap.fill(subtitle[ins['imdb_key']]['lines'][idx], 40) for idx in iid]
#     # print(sentences)
#
#     plt.figure(figsize=(6, int(attn.shape[0] / 2)))
#     ax = sns.heatmap(attn, vmin=0, vmax=1)
#     ax.set_xticklabels([textwrap.fill(ins['question'], 50)] +
#                        [textwrap.fill(ans, 50) for ans in ins['answers']], rotation=45, ha='right')
#     ax.set_yticklabels(sentences, rotation=0)
#     ax.set_title('%d %d' % (pair[0], pair[1]))
#     plt.savefig('fig/%s.jpg' % ins['qid'].replace(':', ''), bbox_inches='tight')
#     plt.close()
#
# def main():
#     train_qa = QA().include(video_clips=True, split={'train'}).get()
#     val_qa = QA().include(video_clips=True, split={'val'}).get()
#
#     with trange(len(train_qa)) as pbar, Pool(8) as pool:
#         for _ in pool.imap_unordered(train_save, train_qa):
#             pbar.update()
#
#     with trange(len(val_qa)) as pbar, Pool(8) as pool:
#         for _ in pool.imap_unordered(val_save, val_qa):
#             pbar.update()
#
# # uniform_data = np.random.rand(3,300)
#
#
# #
# # plt.figure(figsize=(20, 3))
# # uniform_data = np.zeros((3, 300))
# # ax = sns.heatmap(uniform_data, xticklabels=50, yticklabels=['GT','Q','A'], vmin=0, vmax=1)
# #
# #
# # plt.savefig('tests.jpg')
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mod', default='model_se_spec-02-', help='Model used to train.')
#     args = parser.parse_args()
#     index = du.json_load(_mp.sample_index_file)
#     subtitle = Subtitle().get()
#     train_attn = np.load(os.path.join(_mp.attn_dir, args.mod, 'train_attn.npz'))
#     val_attn = np.load(os.path.join(_mp.attn_dir, args.mod, 'val_attn.npz'))
#     train_pair = np.load(os.path.join(_mp.attn_dir, args.mod, 'train_pair.npz'))
#     val_pair = np.load(os.path.join(_mp.attn_dir, args.mod, 'val_pair.npz'))
#     main()

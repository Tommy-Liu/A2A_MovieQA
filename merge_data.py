from functools import partial
from multiprocessing import Pool, Manager
from os.path import join

import numpy as np
from tqdm import tqdm

from config import MovieQAPath
from utils import data_utils as du

_mp = MovieQAPath()


def load_ques(encode_dict, qa):
    encode_dict[qa['qid']] = np.load(join(_mp.encode_dir, qa['qid'] + '.npy'))
    encode_dict[qa['qid'] + 'spec'] = np.load(join(_mp.encode_dir, qa['qid'] + '_spec.npy'))
    encode_dict[qa['qid'] + 'correct_index'] = qa['correct_index']


def load_feat(objfeat_dcit, subtfeat_dict, imdb_key):
    objfeat_dcit[imdb_key] = np.load(join(_mp.object_feature_dir, imdb_key + '.npy'))
    subtfeat_dict[imdb_key] = np.load(join(_mp.encode_dir, imdb_key + '.npy'))


def main():
    manager = Manager()
    encode_dict = manager.dict()
    # objfeat_dcit = manager.dict()
    # subtfeat_dict = manager.dict()
    objfeat_dcit = {}
    subtfeat_dict = {}
    func = partial(load_ques, encode_dict)
    with Pool(4) as pool, \
            tqdm([qa for qa in du.json_load(_mp.qa_file) if qa['video_clips']]) as pbar:
        for _ in pool.imap_unordered(func, pbar):
            pass
    np.savez(_mp.qa_feature, **encode_dict.copy())
    # sample = du.json_load(_mp.sample_frame_file)
    # for imdb_key in tqdm(sample.keys()):
    #     objfeat_dcit[imdb_key] = np.load(join(_mp.object_feature_dir, imdb_key + '.npy'))
    #     subtfeat_dict[imdb_key] = np.load(join(_mp.encode_dir, imdb_key + '.npy'))
    # np.savez(_mp.object_feature, **objfeat_dcit.copy())
    # np.savez(_mp.subtitle_feature, **subtfeat_dict.copy())


if __name__ == '__main__':
    main()

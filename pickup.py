import os
from shutil import copy

import numpy as np

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath
from data.data_loader import QA, Subtitle

# from pprint import pprint

# tt0086190

_mp = MovieQAPath()


def main():
    index = du.json_load(_mp.sample_index_file)
    subtitle = Subtitle().include(imdb_key=['tt0086190']).get()
    sample = du.json_load(_mp.sample_frame_file)
    qa = QA().include(imdb_key=['tt0086190']).get()

    # for ins in qa:
    #     if ins['video_clips']:
    #         print(ins['qid'])
    #         print(ins['question'])
    #         print(ins['answers'])
    #         print(ins['answers'][ins['correct_index']])
    ins = qa[0]
    spec = np.load(os.path.join(_mp.encode_dir, ins['qid'] + '_spec' + '.npy'))
    iid = [idx for i, idx in enumerate(index[ins['imdb_key']]) if spec[i] == 1]
    sentences = [subtitle[ins['imdb_key']]['lines'][idx] for idx in iid]
    imgs = []
    for v in sorted([fu.basename_wo_ext(n) for n in ins['video_clips']]):
        imgs.extend([os.path.join(_mp.image_dir, v, '%s_%05d.jpg' % (v, i + 1))
                     for i in sample[ins['imdb_key']][v]])
    print(len(imgs))
    for idx, img in enumerate(imgs):
        copy(img, os.path.join(_mp.benchmark_dir, 'pickup', '%d_%s.jpg' % (idx, sentences[idx])))
    # ins['lines'] = sentences
    du.json_dump(ins, os.path.join(_mp.benchmark_dir, 'pickup.json'))

    # pprint(sentences)
    # input()
    # print(subtitle['tt0086190']['lines'])


if __name__ == '__main__':
    main()

import argparse
import os
from functools import partial
from glob import glob
from multiprocessing import Pool, Manager
from os.path import join

import imageio
import numpy as np
from tqdm import tqdm

import utils.data_utils as du
from config import MovieQAPath
from data.data_loader import duration
from utils import func_utils as fu

_mp = MovieQAPath()


def check_and_extract_videos(extract, video_clips, video_data, key):
    temp_video_data, delta, img_list = {}, 5, []
    nil_img = np.zeros((299, 299, 3), dtype=np.uint8)

    for video in video_clips[key]:
        del img_list[:]

        base_name = fu.basename_wo_ext(video)
        img_dir = join(_mp.image_dir, base_name)
        extracted = glob(join(img_dir, '*.jpg'))

        try:
            reader = imageio.get_reader(video, ffmpeg_params=['-analyzeduration', '10M'])
        except OSError:
            start, end = duration(base_name)
            num_frame = end - start
            meta_data = {'nframes': num_frame}

            if meta_data['nframes'] > len(extracted) + delta:
                img_list = [nil_img] * num_frame
        else:
            meta_data = reader.get_meta_data()

            if meta_data['nframes'] > len(extracted) + delta:
                try:
                    for img in reader:
                        img_list.append(img)
                except RuntimeError:
                    pass

        meta_data['real_frames'] = len(extracted)
        # Check if already extracted or not
        if img_list:
            if len(extracted) != len(img_list) and extract:
                fu.make_dirs(img_dir)
                for i, img in enumerate(img_list):
                    imageio.imwrite(join(img_dir, '%s_%05d.jpg' % (base_name, i + 1)), img)
            meta_data['real_frames'] = len(img_list)

        temp_video_data[base_name] = meta_data

    video_data[key] = temp_video_data


def get_videos_clips():
    with os.scandir(_mp.video_clips_dir) as it:
        movie_dirs = [entry.path for entry in it if entry.is_dir()]

    video_clips = {}
    for mov in tqdm(movie_dirs, desc='Get video clips'):
        video_clips[os.path.basename(mov)] = glob(join(mov, '*.mp4'))

    return video_clips


def video_process(extract):
    fu.make_dirs(_mp.image_dir)
    manager = Manager()

    video_clips = manager.dict(get_videos_clips())
    video_data = manager.dict()
    keys = list(video_clips.keys())

    with Pool(16) as p, tqdm(total=len(keys), desc="Check and extract videos") as pbar:
        check_func = partial(check_and_extract_videos, extract, video_clips, video_data)

        for _ in p.imap_unordered(check_func, keys):
            pbar.update()

    du.json_dump(video_data.copy(), _mp.video_data_file)


def check():
    video_data = du.json_load(_mp.video_data_file)

    for volume in video_data.values():
        for v in volume:
            img_dir = join(_mp.image_dir, v)
            true_length = len(glob(join(img_dir, '*.jpg')))
            if volume[v]['real_frames'] != len(glob(join(img_dir, '*.jpg'))):
                print(v, true_length, volume[v]['real_frames'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_extract', action='store_false', help='Run without frame extracting.')
    parser.add_argument('--check', action='store_true', help='Checl that number of image is correct')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.check:
        video_process(args.no_extract)
    else:
        check()

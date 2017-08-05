import imageio
import sys
import os
import re

from glob import glob
from tqdm import tqdm
from os.path import join

data_dir = '/home/tommy8054/MovieQA_benchmark/story/video_clips'
matidx_dir = '/home/tommy8054/MovieQA_benchmark/story/matidx'
subt_dir = '/home/tommy8054/MovieQA_benchmark/story/subtt'
metadata = './metadata.json'
video_img = './video_img'

DIR_PATTERN_ = 'tt*'
VIDEO_PATTERN_ = '*.mp4'
CLEAN_PATTERN_ = ''
videos_dirs = [d for d in glob(os.path.join(data_dir, DIR_PATTERN_)) if os.path.isdir(d)]


def clean_token(l):
    """
    Clean up Subrip tags.
    :param l: a string of line
    :return: a cleaned string of line
    """
    return re.sub('<.*?>', '', l)


def get_start_and_end_time(l):
    """
    Get the start time and end time of a line.
    :param l: a string of the time interval
    :return: start time and end time (second [ float ])
    """
    comp = re.split(' --> |:|,', l)
    offset = 0
    start_time = int(comp[offset + 0]) * 3600 + \
                 int(comp[offset + 1]) * 60 + \
                 int(comp[offset + 2]) + \
                 int(comp[offset + 3]) / 1000
    offset = 4
    end_time = int(comp[offset + 0]) * 3600 + \
               int(comp[offset + 1]) * 60 + \
               int(comp[offset + 2]) + \
               int(comp[offset + 3]) / 1000
    return start_time, end_time


def exist_make_dirs(d):
    """
    If the directory dose not exist, make one.
    :param d: a string of directory path.
    :return: None
    """
    if not os.path.exists(d):
        os.makedirs(d)


def get_base_name(p):
    """
    Get the subdirectory or file name
    in the last position of the path p.
    :param p: a string of directory or file path.
    :return: a string of base name.
    """
    return p.split('/')[-1]


def get_base_name_without_ext(p):
    """
    Get the base name without extension
    :param p: a string of directory or file path.
    :return: base name
    """
    base_name = get_base_name(p)
    end = len(base_name.split('.')[-1]) + 1  # + dot
    return base_name[:end]


def get_start_frame(p):
    """
    Get start frame.
    :param p: file path or name.
    :return: an integer of start frame
    """
    return re.split(r'\.|-', p)[-5]


def get_videos_clips():
    """
    Get all video clips path.
    :return: a dictionary with key:imdb, value:video clips path
    """
    videos_clips = {}
    for d in tqdm(videos_dirs):
        imdb = get_base_name(d)
        videos_clips[imdb] = glob(os.path.join(d, VIDEO_PATTERN_))
    return videos_clips


def get_matidx(p):
    """
    Get the mapping of frame and time from the file.
    :param p: file path
    :return: a dictionary with key:frame #, value:time(second [ float ])
    """
    matidx = {}
    with open(join(matidx_dir, p + '.matidx'), 'r') as f:
        for l in tqdm(f):
            comp = l.replace('\n', '').split(' ')
            idx, time = int(comp[0]), float(comp[1])
            matidx[idx] = time
    return matidx


def get_line(line_list, i):
    """
    Get next element and add one to index.
    :param line_list: a list of lines.
    :param i: index
    :return: a string of next line, one-added index
    """
    return line_list[i], i + 1


def map_frame_to_subtitle(matidx, base_name):
    """
    
    :param matidx:
    :param base_name:
    :return:
    """
    with open(join(subt_dir, base_name + '.srt')) as f:
        line_list = list(f)
    line_list = [l.replace('\n', '') for l in line_list]
    i, j = 0, 0
    frame_to_subtitle = {}
    while i < len(line_list):
        l, i = get_line(line_list, i)
        lines = []
        if l.isdigit():
            l, i = get_line(line_list, i)
            start_time, end_time = get_start_and_end_time(l)
            l, i = get_line(line_list, i)
            while l != '':
                lines.append(l)
                l, i = get_line(line_list, i)
            lines = clean_token(' '.join(lines))
            while matidx[j] < end_time:
                frame_to_subtitle[j] = lines
                j = j + 1
    return frame_to_subtitle


def check_and_extract_videos_and_align_subtitle():
    videos_clips = get_videos_clips()
    avail_video_map = {}
    avail_video_list = []
    avail_video_info = {}
    avail_video_subt = {}
    unavail_video_list = []
    for key in tqdm(videos_clips.keys()):
        frame_to_subtitle = map_frame_to_subtitle()
        for video in videos_clips[key]:
            base_name = get_base_name_without_ext(video)
            exist_make_dirs(
                join(video_img,
                     base_name)
            )
            try:
                reader = imageio.get_reader(video)
                img_list = list(reader)
                sys.stdout.write('\r%s\'s frames: %d' % (
                    get_base_name(video), len(img_list)))
                sys.stdout.flush()
            except:
                unavail_video_list.append(video)
                avail_video_map[base_name] = False
            else:
                reader = imageio.get_reader(video)
                avail_video_map[base_name] = True
                avail_video_list.append(base_name)
                avail_video_info[base_name] = reader.get_meta_data()
                start_frame = int(get_start_frame(base_name))
                avail_video_info[base_name]['start_frame'] = start_frame
                img_list = list(reader)
                for img in img_list:
                    pass


class MovieDataset(object):
    def __init__(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()

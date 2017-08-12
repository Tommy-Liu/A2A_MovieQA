import multiprocessing
import traceback
import imageio
import codecs
import json
# import sys
import os
import re


from glob import glob
from tqdm import tqdm
from os.path import join
from functools import partial
from multiprocessing import Pool, Manager

data_dir = '/home/tommy8054/MovieQA_benchmark/story/video_clips'
matidx_dir = '/home/tommy8054/MovieQA_benchmark/story/matidx'
subt_dir = '/home/tommy8054/MovieQA_benchmark/story/subtt'
metadata = './metadata.json'
video_img = './video_img'

DIR_PATTERN_ = 'tt*'
VIDEO_PATTERN_ = '*.mp4'
IMAGE_PATTERN_ = '*.jpg'
videos_dirs = [d for d in glob(os.path.join(data_dir, DIR_PATTERN_)) if os.path.isdir(d)]


def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


def clean_token(l):
    """
    Clean up Subrip tags.
    :param l: a string of line
    :return: a cleaned string of line
    """
    return re.sub(r'<.*?>', '', l)


def get_start_and_end_time(l):
    """
    Get the start time and end time of a line.
    :param l: a string of the time interval
    :return: start time and end time (second [ float ])
    """
    comp = re.split(r' --> |:|,', l)
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


# Fuck os.path.basename. I wrote my own version.
def get_base_name(p):
    """
    Get the subdirectory or file name
    in the last position of the path p.
    :param p: a string of directory or file path.
    :return: a string of base name.
    """
    pos = -1
    if p.split('/')[pos] == '':
        pos = -2
    return p.split('/')[pos]


# Wrapped function
def get_base_name_without_ext(p):
    """
    Get the base name without extension
    :param p: a string of directory or file path.
    :return: base name
    """
    base_name = get_base_name(p)
    base_name = os.path.splitext(base_name)[0]
    return base_name


def get_start_and_end_frame(p):
    """
    Get start and end frame.
    :param p: file path or name.
    :return: 2 integers of start and end frame #.
    """
    comp = re.split(r'\.|-', p)
    return int(comp[-5]), int(comp[-3])


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
    :param p: base name
    :return: a dictionary with key:frame #, value:time(second [ float ])
    """
    matidx = {}
    with open(join(matidx_dir, p + '.matidx'), 'r') as f:
        for l in f:
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
    Map each line of subtitle to the frame.
    :param matidx: a dictionary with key:frame #, value:time(second [ float ])
    :param base_name: imdb name
    :return: a dictionary with key:frame #, value:a string of line.
    """
    try:
        line_list = []
        with codecs.open(join(subt_dir, base_name + '.srt'), 'r',
                         encoding='utf-8', errors='ignore') as f:
            for l in f:
                line_list.append(re.sub('\n|\r', '', l))
            # Some of subtitles don't have last new line.
            # So, we add one.
            if line_list[-1] != '':
                line_list.append('')

        # i for subtitle line #
        # j for frame #
        i, j = 0, 0
        frame_to_subtitle = {}
        while i < len(line_list):
            l, i = get_line(line_list, i)
            lines = []
            # The first line of each line is a digit.
            if l.isdigit():
                l, i = get_line(line_list, i)
                # The second line of each line is time interval.
                start_time, end_time = get_start_and_end_time(l)
                l, i = get_line(line_list, i)
                # Then, the true subtitle lines.
                # When encounter '', stop collect lines.
                while l != '':
                    lines.append(l)
                    l, i = get_line(line_list, i)
                # Clean lines?? good or bad?
                # Cuz it might be another clue.
                # lines = clean_token(' '.join(lines))
                lines = ' '.join(lines)
                # print(lines)
                # Iterate each frame. If in the interval,
                # assign lines to it. If greater than end_time,
                # break the roop. If smaller than start_time,
                # assign empty string.
                while j < len(matidx):
                    if start_time <= matidx[j] <= end_time:
                        frame_to_subtitle[j] = lines
                    elif matidx[j] > end_time:
                        break
                    else:
                        frame_to_subtitle[j] = ''
                    j = j + 1
        # Loop over the rest of frames, and
        # assign them empty string.
        while j < len(matidx):
            frame_to_subtitle[j] = ''
            j = j + 1
    except Exception as e:
        error(traceback.format_exc())
        raise Exception(base_name)
    return frame_to_subtitle


def check_and_extract_videos_and_align_subtitle(videos_clips,
                                                avail_video_map,
                                                avail_video_list,
                                                avail_video_info,
                                                avail_video_subt,
                                                unavail_video_list,
                                                key):
    """

    :return:
    """
    # print('Start %s !' % key)
    frame_to_subtitle = map_frame_to_subtitle(get_matidx(key), key)
    for video in videos_clips[key]:
        base_name = get_base_name_without_ext(video)
        exist_make_dirs(
            join(video_img,
                 base_name)
        )
        reader = imageio.get_reader(video)
        img_list = list(reader)
        print('Start extract %s with %d frames.' %
              (get_base_name(video), len(img_list)))
        video_length = len(img_list)
    #     try:
    #         reader = imageio.get_reader(video)
    #         if len(glob(join(video_img, base_name, IMAGE_PATTERN_))) < len(img_list) - 1:
    #             print('Start extract %s with %d frames.' %
    #                   (get_base_name(video), len(img_list)))
    #             img_list = list(reader)
    #             video_length = len(img_list)
    #         else:
    #             video_length = len(glob(join(video_img, base_name, IMAGE_PATTERN_)))
    #             print('Start process %s.' % (get_base_name(video)))
    #     except:
    #         unavail_video_list.append(video)
    #         avail_video_map[base_name] = False
    #     else:
    #         avail_video_map[base_name] = True
    #         avail_video_list.append(base_name)
    #         avail_video_info[base_name] = reader.get_meta_data()
    #         start_frame, end_frame = get_start_and_end_frame(video)
    #         avail_video_info[base_name]['start_frame'] = start_frame
    #         avail_video_info[base_name]['end_frame'] = end_frame
    #         subt = []
    #         for i in range(video_length):
    #             subt.append(frame_to_subtitle[min([start_frame + i, len(frame_to_subtitle) - 1])])
    #         avail_video_subt[base_name] = subt
    #         exist_make_dirs(join(video_img, base_name))
    #         if len(glob(join(video_img, base_name, IMAGE_PATTERN_))) < len(img_list) - 1:
    #             for i, img in enumerate(img_list):
    #                 imageio.imwrite(join(video_img, base_name, 'img_%05d.jpg' % (i + 1)), img)
        print('%s done.' % get_base_name(video))


## tt0109446.sf-046563.ef-056625.video.mp4
class MovieDataset(object):
    def __init__(self):
        pass


def main():
    videos_clips = get_videos_clips()
    # matidx = get_matidx('tt1058017')
    # frame_to_subtitle = map_frame_to_subtitle(matidx,'tt1058017')
    with Manager() as manager:
        shared_videos_clips = manager.dict(videos_clips)
        shared_avail_video_map = manager.dict()
        shared_avail_video_list = manager.list()
        shared_avail_video_info = manager.dict()
        shared_avail_video_subt = manager.dict()
        shared_unavail_video_list = manager.list()
        keys = list(iter(videos_clips.keys()))
        do_what_i_told_you_to_do = partial(check_and_extract_videos_and_align_subtitle,
                                           shared_videos_clips,
                                           shared_avail_video_map,
                                           shared_avail_video_list,
                                           shared_avail_video_info,
                                           shared_avail_video_subt,
                                           shared_unavail_video_list)
        with Pool(8) as p:
            p.map(do_what_i_told_you_to_do, keys)

    #     avail_video_metadata = {
    #         'map': shared_avail_video_map.copy(),
    #         'list': list(shared_avail_video_list),
    #         'info': shared_avail_video_info.copy(),
    #         'subtitle': shared_avail_video_subt.copy(),
    #         'unavailable': list(shared_unavail_video_list)
    #     }
    # json.dump(avail_video_metadata, open('avail_video_metadata.json', 'w'))



if __name__ == '__main__':
    main()



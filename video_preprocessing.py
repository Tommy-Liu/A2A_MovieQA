import codecs
import json
import multiprocessing
import os
import re
import sys
import traceback
from functools import partial
from glob import glob
from multiprocessing import Pool, Manager
from os.path import join

import imageio
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from data_utils import exist_make_dirs, get_base_name, get_base_name_without_ext, \
    clean_token

data_dir = '/home/tommy8054/MovieQA_benchmark/story/video_clips'
matidx_dir = '/home/tommy8054/MovieQA_benchmark/story/matidx'
subt_dir = '/home/tommy8054/MovieQA_benchmark/story/subtt'
metadata = './metadata.json'
video_img = './video_img'

DIR_PATTERN_ = 'tt*'
VIDEO_PATTERN_ = '*.mp4'
IMAGE_PATTERN_ = '*.jpg'
ALIGN_SUBTITLE_PATTERN_ = '\r>> Align subtitles  %d/%d IMDB: %s'
allow_discard_offset = 10
videos_dirs = [d for d in glob(os.path.join(data_dir, DIR_PATTERN_)) if os.path.isdir(d)]


def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


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


def get_start_and_end_frame(p):
    """
    Get start and end frame.
    :param p: file path or name.
    :return: 2 integers of start and end frame #.
    """
    comp = re.split(r'[.-]', p)
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


def flush_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def map_frame_to_subtitle(imdb_key):
    """
    Map each line of subtitle to the frame.
    :param imdb_key: imdb name
    :return: a dictionary with key:frame #, value:a string of line.
    """
    try:
        matidx = get_matidx(imdb_key)
        line_list = []
        # Read all subtitles from file.
        with codecs.open(join(subt_dir, imdb_key + '.srt'), 'r',
                         encoding='utf-8', errors='ignore') as f:
            for l in f:
                line_list.append(re.sub(r'[\n\r]', '', l))
            # Some of subtitles don't have the last new line.
            # So, we add one.
            if line_list[-1] != '':
                line_list.append('')

        # i for subtitle line number.
        # j for frame number.
        i, j = 0, 0
        frame_to_subtitle = []
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
                # Update: Fuck those tokens.
                lines = word_tokenize(clean_token(' '.join(lines)))
                # Iterate each frame to assign lines to it.
                while j < len(matidx) and matidx[j] <= end_time:
                    # flush_print(ALIGN_SUBTITLE_PATTERN_ % (
                    #     j + 1, len(matidx), imdb_key))
                    if start_time > matidx[j]:
                        frame_to_subtitle.append([])
                    else:
                        frame_to_subtitle.append(lines)
                    j = j + 1
        # Loop over the rest of frames, and
        # assign them empty string.
        while j < len(matidx):
            # flush_print(ALIGN_SUBTITLE_PATTERN_ % (
            #     j + 1, len(matidx), imdb_key))
            frame_to_subtitle.append([])
            j = j + 1
    except Exception:
        error(traceback.format_exc())
        raise Exception(imdb_key)
    return frame_to_subtitle


def align_subtitle(video_clips,
                   avail_video_info,
                   avail_video_subt,
                   avail_video_list,
                   key):
    print(key, 'start!')
    frame_to_subtitle = map_frame_to_subtitle(key)
    for video in video_clips[key]:
        base_name = get_base_name_without_ext(video)
        if base_name in avail_video_list:
            start_frame, end_frame = get_start_and_end_frame(video)
            avail_video_info[base_name]['start_frame'] = start_frame
            avail_video_info[base_name]['end_frame'] = end_frame

            subt = []
            for i in range(avail_video_info[base_name]['real_frames']):
                subt.append(frame_to_subtitle[
                                min([start_frame + i,
                                     len(frame_to_subtitle) - 1])])
            avail_video_subt[base_name] = subt
    print(key, 'done!')


def check_video(video):
    img_list = []
    try:
        base_name = get_base_name_without_ext(video)
        reader = imageio.get_reader(video)
        images = glob(join(video_img, base_name, IMAGE_PATTERN_))
        meta_data = reader.get_meta_data()
        nframes = meta_data['nframes']
        meta_data['real_frames'] = len(images)
        if not (nframes - meta_data['real_frames'] < allow_discard_offset):
            for img in reader:
                img_list.append(img)
        flag = True
    except OSError:
        print(get_base_name(video), 'failed.')
        meta_data = None
        flag = False
    except RuntimeError:
        if nframes - len(img_list) < allow_discard_offset:
            flag = True
        else:
            print(get_base_name(video), 'failed.')
            flag = False
    except:
        print('Something fucked up !!')
        flag = False
        meta_data = None
        raise
    finally:
        return flag, img_list, meta_data


def check_and_extract_videos(videos_clips,
                             avail_video_list,
                             avail_video_info,
                             unavail_video_list,
                             key):
    """

    :return:
    """
    # print('Start %s !' % key)
    for video in videos_clips[key]:
        base_name = get_base_name_without_ext(video)
        flag, img_list, meta_data = check_video(video)

        if flag:
            if len(img_list) > 0:
                exist_make_dirs(join(video_img, base_name))
                for i, img in enumerate(img_list):
                    imageio.imwrite(join(video_img, base_name, 'img_%05d.jpg' % (i + 1)), img)

            avail_video_list.append(base_name)
            avail_video_info[base_name] = meta_data
        else:
            if os.path.exists(join(video_img, base_name)):
                os.system('rm -rf %s' % join(video_img, base_name))
            unavail_video_list.append(video)


# tt0109446.sf-046563.ef-056625.video.mp4
class MovieDataset(object):
    def __init__(self):
        pass


def main():
    videos_clips = get_videos_clips()
    # frame_to_subtitle = map_frame_to_subtitle('tt1058017')
    # print(json.dumps(frame_to_subtitle[10000:10100], indent=4))
    with Manager() as manager:
        # avail_video_metadata = json.load(open('./avail_video_metadata.json', 'r'))
        shared_videos_clips = manager.dict(videos_clips)
        shared_avail_video_list = manager.list()  # avail_video_metadata['list'])
        shared_avail_video_info = manager.dict()  # avail_video_metadata['info'])
        shared_avail_video_subt = manager.dict()
        shared_unavail_video_list = manager.list()  # avail_video_metadata['unavailable'])
        keys = list(iter(videos_clips.keys()))
        check_func = partial(check_and_extract_videos,
                             shared_videos_clips,
                             shared_avail_video_list,
                             shared_avail_video_info,
                             shared_unavail_video_list)
        with Pool(8) as p:
            p.map(check_func, keys)

        avail_video_metadata = {
            'list': list(shared_avail_video_list),
            'info': shared_avail_video_info.copy(),
            'unavailable': list(shared_unavail_video_list)
        }
        json.dump(avail_video_metadata, open('avail_video_metadata.json', 'w'))

        align_func = partial(align_subtitle,
                             shared_videos_clips,
                             shared_avail_video_info,
                             shared_avail_video_subt,
                             shared_avail_video_list)
        with Pool(8) as p:
            p.map(align_func, keys)
        json.dump(shared_avail_video_subt.copy(), open('avail_video_subtitle.json', 'w'))


if __name__ == '__main__':
    main()

import argparse
import codecs
import os
import re
import sys
import ujson as json
from functools import partial
from glob import glob
from multiprocessing import Pool, Manager
from os.path import join

import imageio
import pysrt
from nltk.tokenize.moses import MosesTokenizer
# from nltk.tokenize import word_tokenize, RegexpTokenizer, TweetTokenizer
from tqdm import tqdm

import data_utils as du
from config import MovieQAConfig

config = MovieQAConfig()
data_dir = config.video_clips_dir
matidx_dir = config.matidx_dir
subt_dir = config.subt_dir
video_img = config.video_img_dir

DIR_PATTERN_ = 'tt*'
VIDEO_PATTERN_ = '*.mp4'
IMAGE_PATTERN_ = '*.jpg'
ALIGN_SUBTITLE_PATTERN_ = '\r>> Align subtitles  %d/%d IMDB: %s'
allow_discard_offset = 3
videos_dirs = [d for d in glob(os.path.join(data_dir, DIR_PATTERN_)) if os.path.isdir(d)]

# tokenize_func = word_tokenize
# tokenizer = RegexpTokenizer("[\w']+")
# tokenizer = TweetTokenizer()
tokenizer = MosesTokenizer()
# tokenize_func = tokenizer.tokenize
tokenize_func = partial(tokenizer.tokenize, escape=False)


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
    for d in tqdm(videos_dirs, desc='Get video clips:'):
        imdb = du.get_base_name(d)
        videos_clips[imdb] = glob(os.path.join(d, VIDEO_PATTERN_))
    return videos_clips


def get_matidx(p):
    """
    Get the mapping of frame and time from the file.
    :param p: base name
    :return: a dictionary with key:frame #, value:time(second [ float ])
    """
    matidx = []
    with open(join(matidx_dir, p + '.matidx'), 'r') as f:
        for l in f:
            comp = l.replace('\n', '').split(' ')
            time = float(comp[1])
            matidx.append(time)
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


def map_time_subtitle(imdb_key):
    subs = pysrt.open(join(subt_dir, imdb_key + '.srt'), encoding='iso-8859-1')
    times = []
    subtitles = []
    for sub in subs:
        text = re.sub(r'[\n\r]', ' ', sub.text).lower().strip()
        text = du.clean_token(text).strip()  # .encode('cp1252').decode('cp1252')
        text = tokenize_func(text)  # ''|'<space>' -> []
        if text:
            subtitles.append(text)
            start_time = sub.start.ordinal / 1000
            end_time = sub.end.ordinal / 1000
            times.append((start_time, end_time))
    return times, subtitles


def legacy_map_time_subtitle(imdb_key):
    """
    Map each line of subtitle to the interval of start time and end time.
    :param imdb_key: imdb name
    :return: a list containing tuples: (time tuple: (start time, end time), list of line: [words])
    """
    line_list = []
    # Read all subtitles from file.
    with codecs.open(join(subt_dir, imdb_key + '.srt'), 'r',
                     encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_list.append(re.sub(r'[\n\r]', '', line))
        # Some of subtitles don't have the last new line.
        # So, we add one.
        if line_list[-1] != '':
            line_list.append('')

    # i for subtitle line number.
    # j for frame number.
    i = 0
    time_to_subtitle = []
    while i < len(line_list):
        line, i = get_line(line_list, i)
        lines = []
        # The first line of each line is a digit.
        if line.isdigit():
            line, i = get_line(line_list, i)
            # The second line of each line is time interval.
            start_time, end_time = get_start_and_end_time(line)
            line, i = get_line(line_list, i)
            # Then, the true subtitle lines.
            # When encounter '', stop collect lines.
            while line != '':
                lines.append(line)
                line, i = get_line(line_list, i)
            # Clean lines?? good or bad?
            # Cuz it might be another clue.
            # Update: Fuck those tokens.
            lines = tokenize_func(du.clean_token(' '.join(lines)))
            time_to_subtitle.append(((start_time, end_time), lines))
    return time_to_subtitle


def map_frame_to_subtitle(imdb_key):
    # Time interval to subtitle
    subtitle_time_interval, subtitles = map_time_subtitle(imdb_key)
    # Frames map to times
    matidx = get_matidx(imdb_key)

    frame_to_subtitle = [[] for _ in range(len(matidx))]
    frame_to_subtitle_shot = [0 for _ in range(len(matidx))]
    

def lagacy_map_frame_to_subtitle(imdb_key):
    """
    Map each line of subtitle to the frame.
    :param imdb_key: imdb name
    :return: a dictionary with key:frame #, value:a string of line.
    """
    # Time interval to subtitle
    times, subtitles = map_time_subtitle(imdb_key)
    # Frames map to times
    matidx = get_matidx(imdb_key)

    # Frames map to subtitle

    frame_to_subtitle = [[] for _ in range(len(matidx))]
    frame_to_subtitle_idx = [0 for _ in range(len(matidx))]
    interval = matidx[-1] / len(matidx)
    fts_idx = 1
    for t_idx, time in enumerate(times):
        start_time, end_time = time
        start_time = min(max(start_time, matidx[0]), matidx[-1])
        end_time = max(min(end_time, matidx[-1]), matidx[0])
        start_frame = min(max(int(start_time / interval), 0), len(matidx) - 1)
        end_frame = max(min(int(end_time / interval), len(matidx) - 1), 0)
        # shift index of
        while matidx[start_frame] < start_time:
            start_frame += 1
        while start_frame > 0 and matidx[start_frame - 1] >= start_time:
            start_frame -= 1
        while matidx[end_frame] > end_time:
            end_frame -= 1
        while end_frame < len(matidx) - 1 and matidx[end_frame + 1] <= end_time:
            end_frame += 1
        overlap_idx = []
        for i in range(start_frame, end_frame + 1):
            frame_to_subtitle[i] += subtitles[t_idx]
            if frame_to_subtitle_idx[i] == 0:
                frame_to_subtitle_idx[i] = fts_idx
            else:
                if not frame_to_subtitle_idx[i] in overlap_idx:
                    overlap_idx.append(frame_to_subtitle_idx[i])
                frame_to_subtitle_idx[i] = fts_idx + overlap_idx.index(frame_to_subtitle_idx[i]) + 1
        fts_idx += len(overlap_idx) + 1
    assert len(frame_to_subtitle) == len(frame_to_subtitle_idx) == len(matidx), \
        "Numbers of frames are different %d, %d, %d" % (len(frame_to_subtitle), len(frame_to_subtitle_idx), len(matidx))
    return frame_to_subtitle, frame_to_subtitle_idx, matidx


def align_subtitle(video_clips,
                   video_subtitle,
                   video_data,
                   video_subtitle_index,
                   frame_time,
                   key):
    frame_to_subtitle, frame_to_subtitle_idx, matidx = map_frame_to_subtitle(key)

    for video in video_clips[key]:
        base_name = du.get_base_name_without_ext(video)
        if video_data[base_name]['avail']:
            start_frame, end_frame = get_start_and_end_frame(video)
            video_subtitle[base_name] = {
                'subtitle': [
                    frame_to_subtitle[
                        min(start_frame + i,
                            len(frame_to_subtitle) - 1)]
                    for i in range(video_data[base_name]['info']['num_frames'])
                ],
                'frame_time': [
                    matidx[
                        min(start_frame + i,
                            len(frame_to_subtitle) - 1)]
                    for i in range(video_data[base_name]['info']['num_frames'])
                ],
                'subtitle_index': [
                    frame_to_subtitle_idx[
                        min(start_frame + i,
                            len(frame_to_subtitle) - 1)]
                    for i in range(video_data[base_name]['info']['num_frames'])
                ],
                'shot_boundary': video_data[base_name]['data']['shot_boundary'],
            }
            assert len(video_subtitle[base_name]['subtitle']) == \
                   video_data[base_name]['info']['num_frames'] == \
                   len(video_subtitle[base_name]['shot_boundary']), \
                "Not align! %d %d %d" % (len(video_subtitle[base_name]['subtitle']),
                                         video_data[base_name]['info']['num_frames'],
                                         len(video_subtitle[base_name]['shot_boundary']))
        else:
            video_subtitle[base_name] = {}


def lagacy_align_subtitle(video_clips,
                          video_subtitle,
                          video_data,
                          video_subtitle_index,
                          frame_time,
                          key):
    frame_to_subtitle, frame_to_subtitle_idx, matidx = map_frame_to_subtitle(key)

    for video in video_clips[key]:
        base_name = du.get_base_name_without_ext(video)
        if video_data[base_name]['avail']:
            start_frame, end_frame = get_start_and_end_frame(video)
            video_subtitle[base_name] = {
                'subtitle': [
                    frame_to_subtitle[
                        min(start_frame + i,
                            len(frame_to_subtitle) - 1)]
                    for i in range(video_data[base_name]['info']['num_frames'])
                ],
                'frame_time': [
                    matidx[
                        min(start_frame + i,
                            len(frame_to_subtitle) - 1)]
                    for i in range(video_data[base_name]['info']['num_frames'])
                ],
                'subtitle_index': [
                    frame_to_subtitle_idx[
                        min(start_frame + i,
                            len(frame_to_subtitle) - 1)]
                    for i in range(video_data[base_name]['info']['num_frames'])
                ],
                'shot_boundary': video_data[base_name]['data']['shot_boundary'],
            }
            assert len(video_subtitle[base_name]['subtitle']) == \
                   video_data[base_name]['info']['num_frames'] == \
                   len(video_subtitle[base_name]['shot_boundary']), \
                "Not align! %d %d %d" % (len(video_subtitle[base_name]['subtitle']),
                                         video_data[base_name]['info']['num_frames'],
                                         len(video_subtitle[base_name]['shot_boundary']))
        else:
            video_subtitle[base_name] = {}


def check_video(video):
    # initialize
    img_list = []
    flag = True
    meta_data = None
    nframes = 0
    try:
        base_name = du.get_base_name_without_ext(video)
        reader = imageio.get_reader(video)
        images = glob(join(video_img, base_name, IMAGE_PATTERN_))
        meta_data = reader.get_meta_data()
        nframes = meta_data['nframes']
        meta_data['real_frames'] = len(images)
        if not (nframes - meta_data['real_frames'] < allow_discard_offset):
            for img in reader:
                img_list.append(img)
            meta_data['real_frames'] = len(images)
        flag = True
        assert meta_data['real_frames'], "FUCK FUCK FUCK!!!!"
    except OSError:
        # print(get_base_name(video), 'failed.')
        meta_data = None
        flag = False
    except RuntimeError:
        if nframes - len(img_list) < allow_discard_offset:
            flag = True
        else:
            # print(get_base_name(video), 'failed.')
            flag = False
    finally:
        return flag, img_list, meta_data


def get_shot_boundary(base_name, num_frames):
    with codecs.open(join(config.shot_boundary_dir, base_name + '.sbd'), 'r',
                     encoding='utf-8', errors='ignore') as f:
        sbd = []
        for line in f:
            comp = re.sub(r'[\n\r]', '', line).split(' ')
            # print(comp)
            sbd.append((int(comp[0]), int(comp[1])))
    shot_boundary = []
    i = 0
    for frame_idx in range(num_frames):
        if sbd[i][1] - sbd[0][0] < frame_idx and i < len(sbd) - 1:
            i += 1
        shot_boundary.append(i)
    assert shot_boundary, "Strange... Shot boundary fucked up."
    return shot_boundary


def check_and_extract_videos(videos_clips,
                             video_data,
                             shot_boundary,
                             key):
    """

    :return:
    """
    # print('Start %s !' % key)
    for video in videos_clips[key]:
        base_name = du.get_base_name_without_ext(video)
        flag, img_list, meta_data = check_video(video)
        if flag:
            if len(img_list) > 0:
                du.exist_make_dirs(join(video_img, base_name))
                for i, img in enumerate(img_list):
                    imageio.imwrite(join(video_img, base_name, 'img_%05d.jpg' % (i + 1)), img)
            sbd = get_shot_boundary(base_name, meta_data['real_frames'])
            # print(shot_boundary)
            assert len(sbd) == meta_data['real_frames']
            video_data[base_name] = {
                'avail': True,
                'num_frames': meta_data['real_frames'],
                'image_size': meta_data['size'],
                'fps': meta_data['fps'],
                'duration': meta_data['duration'],
            }
            shot_boundary[base_name] = sbd
        else:
            if os.path.exists(join(video_img, base_name)):
                os.system('rm -rf %s' % join(video_img, base_name))
            video_data[base_name] = {
                'avail': False,
            }
            shot_boundary[base_name] = []


# tt0109446.sf-046563.ef-056625.video.mp4

def video_process(manager, shared_videos_clips, keys):
    shared_video_data = manager.dict()
    shared_shot_boundary = manager.dict()

    check_func = partial(check_and_extract_videos,
                         shared_videos_clips,
                         shared_video_data,
                         shared_shot_boundary)

    with Pool(8) as p, tqdm(total=len(keys), desc="Check and extract videos") as pbar:
        for i, _ in enumerate(p.imap_unordered(check_func, keys)):
            pbar.update()

    du.exist_then_remove(config.video_data_file)
    json.dump(shared_video_data.copy(), open(config.video_data_file, 'w'), indent=4)
    du.exist_then_remove(config.shot_boundary_file)
    json.dump(shared_shot_boundary.copy(), open(config.shot_boundary_file, 'w'))

    return shared_video_data


def subtitle_process(manager, shared_videos_clips, shared_video_data, keys):
    shared_video_subtitle = manager.dict()
    shared_video_subtitle_index = manager.dict()
    shared_frame_time = manager.dict()

    align_func = partial(align_subtitle,
                         shared_videos_clips,
                         shared_video_subtitle,
                         shared_video_data,
                         shared_video_subtitle_index,
                         shared_frame_time)

    with Pool(8) as p, tqdm(total=len(keys), desc="Align subtitle") as pbar:
        for i, _ in enumerate(p.imap_unordered(align_func, keys)):
            pbar.update()

    du.exist_then_remove(config.subtitle_file)
    json.dump(shared_video_subtitle.copy(), open(config.subtitle_file, 'w'), indent=4)


def main():
    with Manager() as manager:
        videos_clips = get_videos_clips()
        shared_videos_clips = manager.dict(videos_clips)
        keys = list(iter(videos_clips.keys()))
        if not args.no_video:
            shared_video_data = video_process(manager, shared_videos_clips, keys)
        else:
            shared_video_data = json.load(open(config.video_data_file, 'r'))

        if not args.no_subt:
            subtitle_process(manager, shared_videos_clips, shared_video_data, keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_video', action='store_true', help='Run without video pre-processing.')
    parser.add_argument('--no_subt', action='store_true', help='Run without subtitle pre-processing.')
    args = parser.parse_args()
    main()

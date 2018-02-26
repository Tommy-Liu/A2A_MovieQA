import re
import time
from datetime import timedelta
from glob import glob
from os.path import exists, join
from unicodedata import normalize

from tqdm import tqdm

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath

mp = MovieQAPath()

RGX_TIMESTAMP_MAGNITUDE_DELIM = r'[,.:]'
RGX_TIMESTAMP = RGX_TIMESTAMP_MAGNITUDE_DELIM.join([r'\d+'] * 4)
RGX_INDEX = r'\d+'
RGX_PROPRIETARY = r'[^\r\n]*'
RGX_CONTENT = r'.*?'
RGX_POSSIBLE_CRLF = r'\r?\n'
RGX_FLOAT = r'[+-]?(\d+([.]\d*)?|[.]\d+)'
RGX_IMDB = r'tt\d+'

SRT_REGEX = re.compile(
    r'({idx})\s*{eof}({ts}) --> ({ts}) ?({proprietary}){eof}({content})'
    r'(?:{eof}|\Z)(?:{eof}|\Z|(?=(?:{idx}\s*{eof}{ts})))'
    r'(?=(?:{idx}\s*{eof}{ts}|\Z))'.format(
        idx=RGX_INDEX,
        ts=RGX_TIMESTAMP,
        proprietary=RGX_PROPRIETARY,
        content=RGX_CONTENT,
        eof=RGX_POSSIBLE_CRLF
    ),  # String form
    re.DOTALL)

FRAME_TIME_REGEX = re.compile(
    r'({idx})\s*({time}){eof}?'.format(
        idx=RGX_INDEX,
        time=RGX_FLOAT,
        eof=RGX_POSSIBLE_CRLF
    ),
    re.DOTALL)

SHOT_BOUNDARY_REGEX = re.compile(
    r'({idx})\s*({idx}){eof}?'.format(
        idx=RGX_INDEX,
        eof=RGX_POSSIBLE_CRLF
    ),
    re.DOTALL)

VIDEO_NAME_REGEX = re.compile(
    r'({imdb})[.]sf-({idx})[.]ef-({idx})[.]video'.format(
        imdb=RGX_IMDB,
        idx=RGX_INDEX
    ),
    re.DOTALL
)


def duration(basename):
    match = VIDEO_NAME_REGEX.match(basename).groups()
    return int(match[1]), int(match[2])


class FrameTime(object):
    def __init__(self):
        if exists(mp.frame_time_file):
            self._frame_time = du.json_load(mp.frame_time_file)
        else:
            self._frame_time = self.process()
        self._inc = {'imdb_key': set(list(self._frame_time.keys()))}

    @staticmethod
    def process():
        frame_time = {}
        frame_time_paths = glob(join(mp.frame_time_dir, '*.matidx'))
        for p in tqdm(frame_time_paths, desc='Process frame time'):
            frame_time[fu.basename_wo_ext(p)] = FrameTime.get_frame_time(p)
        du.json_dump(frame_time, mp.frame_time_file, indent=0)
        return frame_time

    @staticmethod
    def get_frame_time(p):
        times = []
        with open(p, 'r') as f:
            for match in FRAME_TIME_REGEX.finditer(f.read()):
                times.append(float(match.group(2)))  # group 0 is the entire match
        return times

    def reset(self):
        self._inc['imdb_key'] = set(list(self._frame_time.keys()))
        return self

    def include(self, imdb_key=None):
        if imdb_key:
            self._inc['imdb_key'].intersection_update(imdb_key)
        return self

    def exclude(self, imdb_key=None):
        if imdb_key:
            self._inc['imdb_key'].difference_update(imdb_key)
        return self

    def get(self):
        return {k: self._frame_time[k] for k in self._frame_time
                if k in self._inc['imdb_key']}


class Subtitle(object):
    def __init__(self):
        if exists(mp.subtitle_file):
            self._subtitle = du.json_load(mp.subtitle_file)
        else:
            self._subtitle = self.process()
        self._inc = {'imdb_key': set(list(self._subtitle.keys()))}

    @staticmethod
    def process():
        subtitle = {}
        subtitle_paths = glob(join(mp.subtitle_dir, '*.srt'))

        for p in tqdm(subtitle_paths, desc='Process subtitle'):
            basename = fu.basename_wo_ext(p)
            subtitle[basename] = {'lines': [], 'start': [], 'end': []}
            with open(p, 'r', encoding='iso-8859-1') as f:
                for match in SRT_REGEX.finditer(f.read()):
                    raw_index, raw_start, raw_end, proprietary, content = match.groups()

                    content = content.strip()
                    content = re.sub(r'\r\n|\n', ' ', content)
                    content = re.sub(r'<.+?>', '', content, flags=re.DOTALL)
                    content = normalize("NFKD", content)
                    content = content.encode('utf-8').decode('ascii', 'ignore')

                    subtitle[basename]['start'].append(Subtitle.timestamp_to_secs(raw_start))
                    subtitle[basename]['end'].append(Subtitle.timestamp_to_secs(raw_end))
                    subtitle[basename]['lines'].append(content)
        du.json_dump(subtitle, mp.subtitle_file, indent=0)
        return subtitle

    @staticmethod
    def timestamp_to_secs(timestamp):
        hrs, mins, secs, msecs = map(int, re.split(r'[:,]', timestamp))
        return timedelta(hours=hrs, minutes=mins, seconds=secs, milliseconds=msecs).total_seconds()

    def reset(self):
        self._inc['imdb_key'] = set(list(self._subtitle.keys()))
        return self

    def include(self, imdb_key=None):
        if imdb_key:
            self._inc['imdb_key'].intersection_update(imdb_key)
        return self

    def exclude(self, imdb_key=None):
        if imdb_key:
            self._inc['imdb_key'].difference_update(imdb_key)
        return self

    def get(self):
        return {k: self._subtitle[k] for k in self._subtitle
                if k in self._inc['imdb_key']}


class QA(object):
    def __init__(self):
        self._qa = du.json_load(mp.qa_file)
        self._split = du.json_load(mp.splits_file)
        self.video_data = du.json_load(mp.video_data_file)
        self._inc = {'split': set(list(self._split.keys())),
                     'imdb_key': set([k for v in self._split.values() for k in v]),
                     'video_clips': {False}}

    def reset(self):
        self._inc['split'] = set(list(self._split.keys()))
        self._inc['imdb_key'] = set([k for v in self._split.values() for k in v])
        self._inc['video_clips'] = {False}
        return self

    def include(self, split=None, imdb_key=None, video_clips=None):
        if split:
            self._inc['split'].intersection_update(split)
        if imdb_key:
            self._inc['imdb_key'].intersection_update(imdb_key)
        if video_clips is True:
            self._inc['video_clips'] = {video_clips}
        elif video_clips:
            if self._inc['video_clips'] == {True} or \
                    self._inc['video_clips'] == {False}:
                self._inc['video_clips'] = set(video_clips)
            else:
                self._inc['video_clips'].update(video_clips)

        return self

    def exclude(self, split=None, imdb_key=None, video_clips=None):
        if split:
            self._inc['split'].difference_update(split)
        if imdb_key:
            self._inc['imdb_key'].difference_update(imdb_key)
        if video_clips is True:
            self._inc['video_clips'] = set()
        elif video_clips:
            if self._inc['video_clips'] == {True} or \
                    self._inc['video_clips'] == {False}:
                self._inc['video_clips'] = set(
                    [k for v in self.video_data.keys() for k in v]).difference_update(video_clips)
            else:
                self._inc['video_clips'].difference_update(video_clips)

        return self

    def get(self):
        qa = self._qa.copy()
        qa = [ins for ins in qa
              if any(s in ins['qid'] for s in self._inc['split']) and
              ins['imdb_key'] in self._inc['imdb_key']]
        if self._inc['video_clips'] == {True}:
            qa = [ins for ins in qa if ins['video_clips']]
        elif self._inc['video_clips'] != {False}:
            if self._inc['video_clips']:
                qa = [ins for ins in qa if fu.intersect(ins['video_clips'], self._inc['video_clips'])]
            else:
                qa = [ins for ins in qa if not ins['video_clips']]
        return qa


class ShotBoundary(object):
    def __init__(self):
        if exists(mp.shot_boundary_file):
            self._sb = du.json_load(mp.shot_boundary_file)
        else:
            self._sb = self.process()
        self._inc = {'imdb_key': set([k.split('.')[0] for k in self._sb]),
                     'videos': set([k for k in self._sb])}

    @staticmethod
    def process():
        shot_boundary = {}
        sb_paths = glob(join(mp.shot_boundary_dir, '*.sbd'))
        for p in tqdm(sb_paths, desc='Process shot boundary'):
            base_name = fu.basename_wo_ext(p)
            shot_boundary[base_name] = {'start': [], 'end': []}
            with open(p, 'r') as f:
                for match in SHOT_BOUNDARY_REGEX.finditer(f.read()):
                    shot_boundary[base_name]['start'].append(int(match.group(1)))
                    shot_boundary[base_name]['end'].append(int(match.group(2)))

        du.json_dump(shot_boundary, mp.shot_boundary_file)
        return shot_boundary

    def reset(self):
        self._inc['imdb_key'] = set([k.split('.')[0] for k in self._sb])
        self._inc['videos'] = set([k for k in self._sb])
        return self

    def include(self, imdb_key=None, videos=None):
        if imdb_key:
            self._inc['imdb_key'].intersection_update(imdb_key)
        if videos:
            self._inc['videos'].intersection_update(videos)
        return self

    def exclude(self, imdb_key=None, videos=None):
        if imdb_key:
            self._inc['imdb_key'].difference_update(imdb_key)
        if videos:
            self._inc['videos'].difference_update(videos)
        return self

    def get(self):
        sb = self._sb.copy()
        if self._inc['imdb_key']:
            sb = {k: sb[k] for k in sb if any(imdb in k for imdb in self._inc['imdb_key'])}
        if self._inc['videos']:
            sb = {k: sb[k] for k in sb if k in self._inc['videos']}
        return sb


class DataLoader(object):
    def __init__(self):
        pass


def main():
    # start_time = time.time()
    # frame_time = FrameTime()
    # print('%.3f s' % (time.time() - start_time))
    # print(len(frame_time.get()))
    # print(len(frame_time.get(exclude=['tt1371111'])))
    # print(len(frame_time.get(include=['tt1371111'])))
    # start_time = time.time()
    # sub = Subtitle()
    # print('%.3f s' % (time.time() - start_time))
    # print(len(sub.get()))
    # print(len(sub.get(exclude=['tt1371111'])))
    # print(len(sub.get(include=['tt1371111'])))
    start_time = time.time()
    qa = QA()
    print(len(qa.get()))
    print(len(qa.include(imdb_key=['tt1371111']).get()))
    print(len(qa.reset().exclude(imdb_key=['tt1371111']).get()))
    print(len(qa.reset().include(video_clips=True).get()))
    print(len(qa.exclude(imdb_key=['tt1371111']).get()))
    print('%.3f s' % (time.time() - start_time))
    # start_time = time.time()
    # sb = ShotBoundary()
    # print('%.3f s' % (time.time() - start_time))
    # print(len(sb.get()))
    # print(len(sb.get(imdb_key=['tt1371111'])))
    # print(len(sb.get_exclude(imdb_key=['tt1371111'])))
    # s = 'tt2310332.sf-187938.ef-188077.video'
    # match = VIDEO_NAME_REGEX.match(s).groups()
    # print(match)


if __name__ == '__main__':
    main()

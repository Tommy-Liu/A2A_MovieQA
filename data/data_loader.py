import re
import time
from copy import deepcopy
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


class FrameTime(object):
    def __init__(self, inc=None, exc=None):
        if inc:
            self.inc = inc
        else:
            self.inc = {'imdb_key': set()}
        if exc:
            self.exc = exc
        else:
            self.exc = {'imdb_key': set()}

    @staticmethod
    def process():
        frame_time = {}
        frame_time_paths = glob(join(mp.frame_time_dir, '*.matidx'))
        for p in tqdm(frame_time_paths, desc='Process frame time'):
            t = FrameTime.get_frame_time(p)
            frame_time[fu.basename_wo_ext(p)] = t
        du.json_dump(frame_time, mp.frame_time_file, indent=0)
        return frame_time

    @staticmethod
    def get_frame_time(p):
        times = []
        with open(p, 'r') as f:
            for match in FRAME_TIME_REGEX.finditer(f.read()):
                times.append(float(match.group(2)))  # group 0 is the entire match
        return times

    def include(self, imdb_key=None):
        if imdb_key:
            self.inc['imdb_key'].update(imdb_key)
        return deepcopy(self)

    def exclude(self, imdb_key=None):
        if imdb_key:
            self.exc['imdb_key'].update(imdb_key)
        return deepcopy(self)

    def _include(self, frame_time):
        if self.inc['imdb_key']:
            return {k: frame_time[k] for k in frame_time if k in self.inc['imdb_key']}
        else:
            return frame_time

    def _exclude(self, frame_time):
        if self.exc['imdb_key']:
            return {k: frame_time[k] for k in frame_time if k not in self.exc['imdb_key']}
        else:
            return frame_time

    def get(self, frame_time):
        return self._exclude(self._include(frame_time))


class Subtitle(object):
    def __init__(self, inc=None, exc=None):
        if inc:
            self.inc = inc
        else:
            self.inc = {'imdb_key': set()}
        if inc:
            self.exc = exc
        else:
            self.exc = {'imdb_key': set()}

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

    def include(self, imdb_key=None):
        if imdb_key:
            self.inc['imdb_key'].update(imdb_key)
        return deepcopy(self)

    def exclude(self, imdb_key=None):
        if imdb_key:
            self.exc['imdb_key'].update(imdb_key)
        return deepcopy(self)

    def _include(self, subtitle):
        if self.inc['imdb_key']:
            return {k: subtitle[k] for k in subtitle if k in self.inc['imdb_key']}
        return self

    def _exclude(self, subtitle):
        if self.exc['imdb_key']:
            return {k: subtitle[k] for k in subtitle if k not in self.exc['imdb_key']}

    def get(self, subtitle):
        return self._exclude(self._include(subtitle))


class QA(object):
    def __init__(self, inc=None, exc=None):
        if inc:
            self.inc = inc
        else:
            self.inc = {'split': set(), 'imdb_key': set(), 'video_clips': set()}
        if exc:
            self.exc = exc
        else:
            self.exc = {'split': set(), 'imdb_key': set(), 'video_clips': set()}

    def include(self, split=None, imdb_key=None, video_clips=None):
        if split:
            self.inc['split'].update(split)
        if imdb_key:
            self.inc['imdb_key'].update(imdb_key)
        if video_clips is True:
            self.inc['video_clips'].update([video_clips])
        elif video_clips:
            self.inc['video_clips'].update(video_clips)
        return deepcopy(self)

    def exclude(self, split=None, imdb_key=None, video_clips=None):
        if split:
            self.exc['split'].update(split)
        if imdb_key:
            self.exc['imdb_key'].update(imdb_key)
        if video_clips is True:
            self.exc['video_clips'].update([video_clips])
        elif video_clips:
            self.exc['video_clips'].update(video_clips)
        return deepcopy(self)

    def _include(self, qa):
        if self.inc['split']:
            qa = [ins for ins in qa if any(s in ins['qid'] for s in self.inc['split'])]
        if self.inc['imdb_key']:
            qa = [ins for ins in qa if ins['imdb_key'] in self.inc['imdb_key']]
        if self.inc['video_clips'] == {True}:
            qa = [ins for ins in qa if ins['video_clips']]
        elif self.inc['video_clips']:
            qa = [ins for ins in qa if fu.intersect(ins['video_clips'], self.inc['video_clips'])]
        return qa

    def _exclude(self, qa):
        if self.exc['split']:
            qa = [ins for ins in qa if not any(s in ins['qid'] for s in self.exc['split'])]
        if self.exc['imdb_key']:
            qa = [ins for ins in qa if ins['imdb_key'] not in self.exc['imdb_key']]
        if self.exc['video_clips'] == {True}:
            qa = [ins for ins in qa if not ins['video_clips']]
        elif self.exc['video_clips']:
            qa = [ins for ins in qa if not fu.intersect(ins['video_clips'], self.exc['video_clips'])]
        return qa

    def get(self, qa):
        return self._exclude(self._include(qa))


class ShotBoundary(object):
    def __init__(self):
        self.inc = {'imdb_key': set(), 'videos': set()}
        self.exc = {'imdb_key': set(), 'videos': set()}

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

    def include(self, imdb_key=None, videos=None):
        if imdb_key:
            self.inc['imdb_key'].update(imdb_key)
        if videos:
            self.inc['videos'].update(videos)
        return deepcopy(self)

    def exclude(self, imdb_key=None, videos=None):
        if imdb_key:
            self.exc['imdb_key'].update(imdb_key)
        if videos:
            self.exc['videos'].update(videos)
        return deepcopy(self)

    def _include(self, shot_boundary):
        if self.inc['imdb_key']:
            shot_boundary = {k: shot_boundary[k] for k in shot_boundary
                             if any(imdb in k for imdb in self.inc['imdb_key'])}
        if self.inc['videos']:
            shot_boundary = {k: shot_boundary[k] for k in shot_boundary if k in self.inc['videos']}
        return shot_boundary

    def _exclude(self, shot_boundary):
        if self.exc['imdb_key']:
            shot_boundary = {k: shot_boundary[k] for k in shot_boundary
                             if not any(imdb in k for imdb in self.exc['imdb_key'])}
        if self.exc['videos']:
            shot_boundary = {k: shot_boundary[k] for k in shot_boundary if k not in self.exc['videos']}
        return shot_boundary

    def get(self, shot_boundary):
        return self._exclude(self._include(shot_boundary))


class DataLoader(object):
    def __init__(self):
        pass

    def __getitem__(self, item):
        if item == 'frame_time':
            if exists(mp.frame_time_file):
                return du.json_load(mp.frame_time_file)
            else:
                return FrameTime.process()
        elif item == 'subtitle':
            if exists(mp.subtitle_file):
                return du.json_load(mp.subtitle_file)
            else:
                return Subtitle.process()
        elif item == 'qa':
            return du.json_load(mp.qa_file)
        elif item == 'shot_boundary':
            if exists(mp.shot_boundary_file):
                return du.json_load(mp.shot_boundary_file)
            else:
                return ShotBoundary.process()


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
    # start_time = time.time()
    # qa = QA()
    # print('%.3f s' % (time.time() - start_time))
    # print(len(qa.get()))
    # print(len(qa.get(imdb_key=['tt1371111'])))
    # print(len(qa.get_exclude(imdb_key=['tt1371111'])))
    # print(len(qa.get(video_clips=True)))
    # print(len(qa.get_exclude(qa.get(video_clips=True), imdb_key=['tt1371111'])))
    start_time = time.time()
    sb = ShotBoundary()
    print('%.3f s' % (time.time() - start_time))
    print(len(sb.get()))
    print(len(sb.get(imdb_key=['tt1371111'])))
    print(len(sb.get_exclude(imdb_key=['tt1371111'])))


if __name__ == '__main__':
    main()

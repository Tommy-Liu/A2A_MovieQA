import re
from datetime import timedelta
from glob import glob
from os.path import exists, join
from unicodedata import normalize

import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm

import utils.data_utils as du
import utils.func_utils as fu
from config import MovieQAPath

_mp = MovieQAPath()

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
    """
    A simple data loader for loading frame time.
    self._frame_time =
    {
        'ttxxxxxx':[0.xxx, 1.xxx, ..., 14938.xxx],
        ...,
    }
    ttxxxxx: imdb key
    [0.xxx, 1.xxx, ..., 14938.xxx]: a list of timestamp of each frame

    self._inc =
    {
        'imdb_key': set of included imdb keys
    }
    """
    def __init__(self):
        if exists(_mp.frame_time_file):
            self._frame_time = du.json_load(_mp.frame_time_file)
        else:
            self._frame_time = self.process()
        self._inc = {'imdb_key': set(list(self._frame_time.keys()))}

    @staticmethod
    def process():
        """
        Process frame time of each movie, and return a dictionary {imdb_key: a list of timestamp}
        :return frame_time: dictionary mapping imdb key to a list of timestamp
        """
        frame_time = {}
        frame_time_paths = glob(join(_mp.frame_time_dir, '*.matidx'))
        for p in tqdm(frame_time_paths, desc='Process frame time'):
            # fu.basename_wo_ext(p) -> imdb_key
            frame_time[fu.basename_wo_ext(p)] = FrameTime.get_frame_time(p)
        du.json_dump(frame_time, _mp.frame_time_file, indent=0)
        return frame_time

    @staticmethod
    def get_frame_time(p):
        """
        Parse the frame time from the file, and return timestamp of each frame.
        :param p: file path of frame time
        :return times: a list of timestamp in seconds
        """
        times = []
        with open(p, 'r') as f:
            for match in FRAME_TIME_REGEX.finditer(f.read()):
                times.append(float(match.group(2)))  # group 0 is the entire match
        return times

    def reset(self):
        """
        Reset included set to all.
        :return self: self object
        """
        self._inc['imdb_key'] = set(list(self._frame_time.keys()))
        return self

    def include(self, imdb_key=None):
        """
        Intersect the set of imdb_key.
        :param imdb_key: included keys
        :return self: self object
        """
        if imdb_key:
            self._inc['imdb_key'].intersection_update(imdb_key)
        return self

    def exclude(self, imdb_key=None):
        """
        Difference the set of imdb_key.
        :param imdb_key: excluded keys
        :return self: self object
        """
        if imdb_key:
            self._inc['imdb_key'].difference_update(imdb_key)
        return self

    def get(self):
        """
        Return the dictionary mapping imdb key to a list of timestamp confined in self._inc
        :return: dictionary mapping imdb key to a list of timestamp
        """
        return {k: self._frame_time[k] for k in self._frame_time
                if k in self._inc['imdb_key']}


class Subtitle(object):
    """
        A simple data loader for loading frame time.
        self._subtitle =
        {
            'ttxxxxxx':{
                'lines': ['xxxx', 'yyyyyy', ..., 'zzzzzzz'],
                'start': [0.xxx, 1.xxx, ..., 14938.xxx],
                'end': [0.xxx, 1.xxx, ..., 14938.xxx],
            },
            ...,
        }
        ttxxxxx: imdb key
        lines: subtitle sentences
        start: start timestamp of sentences
        end: end timestamp of sentences

        self._inc =
        {
            'imdb_key': set of included imdb keys
        }
        """
    def __init__(self):
        if exists(_mp.subtitle_file):
            self._subtitle = du.json_load(_mp.subtitle_file)
        else:
            self._subtitle = self.process()
        self._inc = {'imdb_key': set(list(self._subtitle.keys()))}

    @staticmethod
    def process():
        """
        Process subtitle files of movies. It will encode the subtitle with ISO-8859-1,
        and substitute new line or <> tokens with '\b' or '', and normalize the characters.
        :return subtitle: dictionary mapping imdb key to subtitle
        """
        subtitle = {}
        # print(_mp.subtitle_dir)
        subtitle_paths = glob(join(_mp.subtitle_dir, '*.srt'))
        # print(subtitle_paths)
        for p in tqdm(subtitle_paths, desc='Process subtitle'):
            iid = 0
            # basename imdb_key
            basename = fu.basename_wo_ext(p)
            subtitle[basename] = {'lines': [], 'start': [], 'end': []}
            with open(p, 'r', encoding='iso-8859-1') as f:
                for match in SRT_REGEX.finditer(f.read()):
                    raw_index, raw_start, raw_end, proprietary, content = match.groups()

                    content = re.sub(r'\r\n|\n', ' ', content)
                    content = re.sub(r'<.+?>', '', content, flags=re.DOTALL)
                    content = re.sub(r'[<>]', '', content)
                    content = normalize("NFKD", content)
                    content = content.encode('utf-8').decode('ascii', 'ignore').strip()

                    if content:
                        content = sent_tokenize(content)
                        content = [sent.strip() for sent in content if sent.strip()]
                        s = Subtitle.timestamp_to_secs(raw_start)
                        e = Subtitle.timestamp_to_secs(raw_end)
                        if s > e:
                            s, e = e, s
                        time_span = (e - s) / len(content)
                        for idx, sent in enumerate(content):
                            subtitle[basename]['start'].append(s + time_span * idx)
                            subtitle[basename]['end'].append(s + time_span * (idx + 1))
                            subtitle[basename]['lines'].append(sent)
                    iid += 1
            index = np.argsort(np.array(subtitle[basename]['start']))
            subtitle[basename]['start'] = [subtitle[basename]['start'][idx] for idx in index]
            subtitle[basename]['end'] = [subtitle[basename]['end'][idx] for idx in index]
            subtitle[basename]['lines'] = [subtitle[basename]['lines'][idx] for idx in index]

        du.json_dump(subtitle, _mp.subtitle_file, indent=0)
        return subtitle

    @staticmethod
    def timestamp_to_secs(timestamp):
        """
        String timestamp to seconds.
        :param timestamp:
        :return: seconds
        """
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
        self._qa = du.json_load(_mp.qa_file)
        self._split = du.json_load(_mp.splits_file)
        self.video_data = du.json_load(_mp.video_data_file)
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
        if exists(_mp.shot_boundary_file):
            self._sb = du.json_load(_mp.shot_boundary_file)
        else:
            self._sb = self.process()
        self._inc = {'imdb_key': set([k.split('.')[0] for k in self._sb]),
                     'videos': set([k for k in self._sb])}

    @staticmethod
    def process():
        shot_boundary = {}
        sb_paths = glob(join(_mp.shot_boundary_dir, '*.sbd'))
        for p in tqdm(sb_paths, desc='Process shot boundary'):
            base_name = fu.basename_wo_ext(p)
            shot_boundary[base_name] = {'start': [], 'end': []}
            with open(p, 'r') as f:
                for match in SHOT_BOUNDARY_REGEX.finditer(f.read()):
                    shot_boundary[base_name]['start'].append(int(match.group(1)))
                    shot_boundary[base_name]['end'].append(int(match.group(2)))

        du.json_dump(shot_boundary, _mp.shot_boundary_file)
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
    subtitle = Subtitle()


if __name__ == '__main__':
    main()

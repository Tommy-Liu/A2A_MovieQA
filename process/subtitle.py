from __future__ import division
from __future__ import print_function

import inspect
import re
from datetime import timedelta
from functools import total_ordering, wraps
from glob import glob
from unicodedata import normalize

import utils.func_utils as fu

RGX_TIMESTAMP_MAGNITUDE_DELIM = r'[,.:]'
RGX_TIMESTAMP = RGX_TIMESTAMP_MAGNITUDE_DELIM.join([r'\d+'] * 4)
RGX_INDEX = r'\d+'
RGX_PROPRIETARY = r'[^\r\n]*'
RGX_CONTENT = r'.*?'
RGX_POSSIBLE_CRLF = r'\r?\n'

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
    re.DOTALL,
)

SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60
HOURS_IN_DAY = 24
MICROSECONDS_IN_MILLISECOND = 1000


def timedelta_to_srt_timestamp(timedelta_timestamp):
    hrs, secs_remainder = divmod(timedelta_timestamp.seconds, SECONDS_IN_HOUR)
    hrs += timedelta_timestamp.days * HOURS_IN_DAY
    mins, secs = divmod(secs_remainder, SECONDS_IN_MINUTE)
    msecs = timedelta_timestamp.microseconds // MICROSECONDS_IN_MILLISECOND
    return '%02d:%02d:%02d,%03d' % (hrs, mins, secs, msecs)


def srt_timestamp_to_timedelta(ts):
    hrs, mins, secs, msecs = map(int, re.split(r'[:,]', ts))
    return timedelta(hours=hrs, minutes=mins, seconds=secs, milliseconds=msecs)


def initializer(func):
    fullargspec = inspect.getfullargspec(func)
    names, varargs, keywords, defaults, kwonlyargs, kwonlydefaults, annotations = fullargspec

    @wraps(func)
    def wrapper(self, *args, **kargs):
        inspect.signature(func).bind(self, *args, **kargs)
        for name, arg in list(zip(names[1:], args)) + list(zip(kargs.keys(), kargs.values())):
            print(name, arg)
            setattr(self, name, arg)
        if defaults:
            for name, default in zip(reversed(names), reversed(defaults)):
                if not hasattr(self, name):
                    setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


@total_ordering
class Line(object):
    @initializer
    def __init__(self, index, start, end, context, decode):
        pass

    def __hash__(self):
        return hash(frozenset(vars(self).items()))

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __lt__(self, other):
        return self.start < other.start or (
                self.start == other.start and self.end < other.end
        )

    def __repr__(self):
        # Python 2/3 cross compatibility
        var_items = getattr(
            vars(self), 'iteritems', getattr(vars(self), 'items')
        )
        item_list = ', '.join(
            '%s=%r' % (k, v) for k, v in var_items()
        )
        return "%s(%s)" % (type(self).__name__, item_list)

    @property
    def srt(self):
        return '{idx}{eol}{start} --> {end}{eol}{content}{eol}{eol}'.format(
            idx=self.index, start=timedelta_to_srt_timestamp(self.start),
            end=timedelta_to_srt_timestamp(self.end),
            content=self.content, eol='\n',
        )


# pos(False): start, pos(True): end
def binary_search(a, v, pos):
    upper = len(a) - 1
    lower = 0
    pivot = (upper + lower) // 2

    while lower < upper:
        if a[pivot] < v:
            lower = pivot + 1
        elif a[pivot] > v:
            upper = pivot - 1
        else:
            return pivot
        pivot = (upper + lower) // 2

    if pos and a[pivot] > v:
        return pivot - 1
    elif not pos and a[pivot] < v:
        return pivot + 1
    else:
        return pivot


class Subtitle(object):
    def __init__(self, srt_file):
        self.lines = []
        self.start = []
        self.end = []
        # self.times
        self.key = fu.basename_wo_ext(srt_file)
        with open(srt_file, 'r', encoding='iso-8859-1') as f:
            for match in SRT_REGEX.finditer(f.read()):
                raw_index, raw_start, raw_end, proprietary, content = match.groups()

                content = content.strip()
                content = re.sub(r'\r\n|\n', ' ', content)
                content = re.sub(r'<.+?>', '', content, flags=re.DOTALL)
                content = normalize("NFKD", content)

                self.start.append(srt_timestamp_to_timedelta(raw_start))
                self.end.append(srt_timestamp_to_timedelta(raw_end))
                self.lines.append(content)


def main():
    subt_list = glob('/mnt/data/tommy8054/MovieQA_benchmark/story/subtt/*.srt')

    #     lines = []
    #     with open(srt_file, 'rb') as f:
    #         for match in SRT_REGEX.finditer(f.read()):
    #             pass
    # print(match.groups())
    # with open
    sub = Subtitle('/home/tommy8054/MovieQA_benchmark/story/subtt/tt0449088.srt')
    # print(sub.lines[0])


if __name__ == '__main__':
    main()

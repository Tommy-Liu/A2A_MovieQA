import tensorflow as tf
import os
from glob import glob
import pprint
import ujson as json

pp = pprint.PrettyPrinter(indent=4, compact=False)


class SummaryHandler(object):
    PATTERN = 'events.out.tfevents.*'

    def __init__(self, path):
        '''
        - EXP path - run1 - event
                   - run2 - event
                   - run3 - event
        :param path: Log path
        '''
        paths = {entry.name: sorted(glob(os.path.join(entry.path, self.PATTERN))[0], reverse=True)
                 for entry in os.scandir(path) if entry.is_dir()}


if __name__ == '__main__':
    sh = SummaryHandler('./log/embedding_log/')

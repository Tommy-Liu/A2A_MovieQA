import os
import re

from config import MovieQAPath
from model.basic_hp import BasicHP


class BasicModel(object):
    def __init__(self):
        self._hp = BasicHP()
        self._mp = MovieQAPath()

    def __str__(self):
        return '.'.join(re.split(r'[./]', __file__)[-3:-1] +
                        ([str(self._hp)] if str(self._hp) != '' else []))

    def __repr__(self):
        return self.__str__()

    @property
    def _log_dir(self):
        return os.path.join(self._mp.log_dir, self.__str__())

    @property
    def _checkpoint_dir(self):
        return os.path.join(self._mp.checkpoint_dir, self.__str__())

    @property
    def _checkpoint_file(self):
        return os.path.join(self._checkpoint_dir, self.__str__())

    @property
    def _best_checkpoint(self):
        return os.path.join(self._checkpoint_dir, 'best', self.__str__())

    @property
    def _attn_dir(self):
        return os.path.join(self._mp.attn_dir, self.__str__())

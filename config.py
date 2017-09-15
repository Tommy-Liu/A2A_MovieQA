from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import json
import os
from contextlib import contextmanager
from glob import glob


class Config(object):
    pass


class MovieQAConfig(Config):
    """Wrapper class for all hyperparameters."""
    NPY_PATTERN_ = '*.npy'
    _group_name = None

    def __init__(self):
        # Directory of data
        with self._create_group('directory'):
            self.data_dir = '../MovieQA_benchmark/story/video_clips'
            self.matidx_dir = '../MovieQA_benchmark/story/matidx'
            self.subt_dir = '../MovieQA_benchmark/story/subtt'
            self.video_img_dir = './video_img'
            self.feature_dir = './features'
            self.dataset_dir = './dataset'
            self.checkpoint_dir = './checkpoint'
            self.log_dir = './log'
        # File names
        with self._create_group('file_names'):
            self.avail_video_metadata_file = './avail_video_metadata.json'
            self.avail_video_subtitle_file = './avail_video_subtitle.json'
            self.avail_split_qa_file = './avail_split_qa.json'
            self.avail_tokenize_qa_file = './avail_tokenize_qa.json'
            self.avail_encode_qa_file = './encode_qa.json'
            self.sep_vocab_file = './avail_separate_vocab.json'
            self.all_vocab_file = './avail_all_vocab.json'
            self.info_file = './info.json'
            self.qa_file = '../MovieQA_benchmark/data/qa.json'
            self.movies_file = '../MovieQA_benchmark/data/movies.json'
            self.splits_file = '../MovieQA_benchmark/data/splits.json'
            self.npy_files = glob(os.path.join(self.feature_dir, self.NPY_PATTERN_))

        # Names
        self.dataset_name = 'movieqa'

        # Tfrecord setting
        self.num_shards = 128

        # Language pre-process
        self.UNK = 'UNK'
        # Training parameter
        self.batch_size = 2

        # Model parameter
        self.feature_dim = 1536

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        self.num_epochs = 20

        self.size_vocab_q = 0
        self.size_vocab_a = 0
        self.size_vocab_s = 0

        self.load_vocab_size()

        self.num_worker = 4

        self.num_training_train_examples = 0
        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5

        with self._create_group('tunable_parameter'):
            self.min_filter_size = 3
            self.max_filter_size = 5

            self.sliding_dim = 1024
            # LSTM input and output dimensionality, respectively.
            self.embedding_size = 512
            self.num_lstm_units = 512

            # If < 1.0, the dropout keep probability applied to LSTM variables.
            self.lstm_dropout_keep_prob = 0.7

            # Optimizer for training the model.
            self.optimizer = "Adam"

            # Learning rate for the initial phase of training.
            self.initial_learning_rate = 0.002
            self.learning_rate_decay_factor = 0.87
            self.num_epochs_per_decay = 1.0

            # If not None, clip gradients to this value.
            self.clip_gradients = 5.0

        self.filter_sizes = list(range(self.min_filter_size,
                                       self.max_filter_size + 1))
        self.load_info()

    @contextmanager
    def _create_group(self, group_name):
        super().__setattr__(group_name, Config())
        super().__setattr__('_group_name', group_name)
        yield
        super().__setattr__('_group_name', None)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if self._group_name:
            super().__getattribute__(self._group_name).__setattr__(key, value)

    def load_vocab_size(self):
        # if not os.path.exists(self.info_file):
        vocab_sep = json.load(open(self.sep_vocab_file, 'r'))
        self.size_vocab_q = len(vocab_sep['vocab_q'])
        self.size_vocab_a = len(vocab_sep['vocab_a'])
        self.size_vocab_s = len(vocab_sep['vocab_s'])
        #     json.dump({
        #         'size_vocab_q': self.size_vocab_q,
        #         'size_vocab_a': self.size_vocab_a,
        #         'size_vocab_s': self.size_vocab_s,
        #     }, open(self.info_file, 'w'))
        # else:
        #     info = json.load(open(self.info_file, 'r'))
        #     self.size_vocab_q = info['size_vocab_q']
        #     self.size_vocab_a = info['size_vocab_a']
        #     self.size_vocab_s = info['size_vocab_s']

    def load_info(self):
        info = json.load(open(self.info_file, 'r'))
        self.__dict__.update(info)

    def update_info(self, item=None, file=None, keys=None):
        if os.path.exists(self.info_file):
            info = json.load(open(self.info_file, 'r'))
        else:
            info = {}
        if file:
            item = json.load(open(file, 'r'))
        if keys:
            for key in keys:
                info[key] = item[key]
                self.__setattr__(key, item[key])
        else:
            info.update(item)
            self.__dict__.update(item)
        json.dump(info, open(self.info_file, 'w'))


def main():
    model_config = MovieQAConfig()
    print(json.dumps(model_config.tunable_parameter.__dict__, indent=4))
    h = hashlib.new('ripemd160')
    h.update(json.dumps(model_config.tunable_parameter.__dict__, indent=4).encode())
    print(h.hexdigest())
    print(model_config.num_training_train_examples)


if __name__ == '__main__':
    main()

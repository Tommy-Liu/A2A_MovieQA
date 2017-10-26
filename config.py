from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import json
import os
from contextlib import contextmanager
from glob import glob
from os.path import join


class Config(object):
    pass


class MovieQAConfig(Config):
    """Wrapper class for all hyperparameters."""
    NPY_PATTERN_ = '*.npy'
    DATASET_PATTERN_ = '%s%s_%s_%s'  # 1. is_training 2. datasetname 3. split 4. modality
    TFRECORD_PATTERN_ = DATASET_PATTERN_ + '_%05d-of-%05d.tfrecord'
    TFRECORD_FILE_PATTERN_ = TFRECORD_PATTERN_.replace('%05d-of-', '*')
    NUMEXAMPLE_PATTERN_ = 'num_' + DATASET_PATTERN_ + '_examples'
    _group_name = None

    def __init__(self):
        # Directory of data
        self.directory = Config()
        with self._create_group('directory'):
            self.movieqa_benchmark_dir = '../MovieQA_benchmark'
            # Directory of original data
            self.video_clips_dir = join(self.movieqa_benchmark_dir, 'story/video_clips')
            self.matidx_dir = join(self.movieqa_benchmark_dir, 'story/matidx')
            self.subt_dir = join(self.movieqa_benchmark_dir, 'story/subtt')
            self.shot_boundary_dir = join(self.movieqa_benchmark_dir, 'story/shot_boundaries')

            # Directory of processed data
            self.data_dir = './data'
            # Directory of all images of video clips
            self.video_img_dir = join(self.data_dir, 'video_img')
            # Directory of all features of video clips
            self.feature_dir = join(self.data_dir, 'features')
            # Directory of tfrecords
            self.dataset_dir = join(self.data_dir, 'dataset')

            # Experiment directory
            self.checkpoint_dir = './checkpoint'
            self.log_dir = './log'
            self.exp_dir = './exp'

        # File names
        self.file_names = Config()
        with self._create_group('file_names'):
            # Video data
            self.video_data_file = join(self.data_dir, 'video_data.json')
            # Shot boundary
            self.shot_boundary_file = join(self.data_dir, 'shot_boundary.json')
            # Subtitle data
            self.subtitle_file = join(self.data_dir, 'video_subtitle.json')
            # Time to frame
            self.frame_time_file = join(self.data_dir, 'frame_time.json')
            # Subtitle shot boundary
            self.subtitle_shot_file = join(self.data_dir, 'video_subtitle_shot.json')
            # Encoded subtitle
            self.encode_subtitle_file = join(self.data_dir, 'encode_subtitle.json')

            # All qas of all splits
            self.total_split_qa_file = join(self.data_dir, 'total_split_qa.json')
            # Available tokenize qa
            self.avail_tokenize_qa_file = join(self.data_dir, 'avail_tokenize_qa.json')
            # Available encoded qa
            self.avail_encode_qa_file = join(self.data_dir, 'avail_encode_qa.json')
            # Training vocabulary
            self.all_vocab_file = join(self.data_dir, 'avail_all_vocab.json')
            # Important information
            self.info_file = join(self.data_dir, 'info.json')
            # Experiment
            self.exp_file = join(self.exp_dir, './exp.json')
            # Original qa
            self.qa_file = join(self.movieqa_benchmark_dir, 'data/qa.json')
            # Original movie infomation
            self.movies_file = join(self.movieqa_benchmark_dir, 'data/movies.json')
            # Original qa split
            self.splits_file = join(self.movieqa_benchmark_dir, 'data/splits.json')
            # Npy file pattern
            self.npy_files = glob(os.path.join(self.feature_dir, self.NPY_PATTERN_))

            # Word embedding, Vocabulary file
            self.fasttext_file = join(self.data_dir, 'crawl-300d-2M.vec')
            self.word2vec_file = join(self.data_dir, 'GoogleNews-vectors-negative300.txt')
            self.glove_file = join(self.data_dir, 'glove.840B.300d.w2v.txt')

            self.w2v_embedding_file = join(self.data_dir, 'w2v_embedding.json')
            self.w2v_embedding_npy_file = join(self.data_dir, 'w2v_embedding.npy')
            self.ft_embedding_file = join(self.data_dir, 'ft_embedding.json')
            self.ft_embedding_npy_file = join(self.data_dir, 'ft_embedding.npy')
            self.glove_embedding_file = join(self.data_dir, 'globe_embedding.json')
            self.glove_embedding_npy_file = join(self.data_dir, 'globe_embedding.npy')

        # Names
        self.dataset_name = 'movieqa'

        # Modality configuration
        self.modality_config = {
            'fixed_num': 105,
            'fixed_interval': 5,
            'shot_major': 4,
            'subtitle_major': 4,
        }

        # Sequences max length
        self.subt_max_length = 41
        self.ans_max_length = 34
        self.ques_max_length = 25

        # Vocabulary frequency threshold
        self.vocab_thr = 0

        # Size of vocabulary
        self.size_vocab = 0

        # Tfrecord setting
        self.num_shards = 128

        # Language pre-process
        self.UNK = 'UNK'
        # Training parameter
        self.batch_size = 2
        # Train shuffle buffer size
        self.size_shuffle_buffer = 64
        # Model parameter
        self.feature_dim = 1536

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # Load vocabulary size
        self.load_vocab_size()

        self.num_training_train_examples = 0
        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5

        self.tunable_parameter = Config()
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

            # Number of sliding convolution layer
            self.num_layers = 1
            # Learning rate for the initial phase of training.
            self.initial_learning_rate = 0.0001
            self.learning_rate_decay_factor = 0.87
            self.num_epochs_per_decay = 1.0

            # If not None, clip gradients to this value.
            self.clip_gradients = 1.0
            # Default modality
            self.modality = 'fixed_num'
            # Number of epochs
            self.num_epochs = 20

        self.filter_sizes = list(range(self.min_filter_size,
                                       self.max_filter_size + 1))
        self.info = {}
        self.load_info()

    def get_num_example(self, split='train', modality='fixed_num',
                        is_training=False):
        return self.info[self.NUMEXAMPLE_PATTERN_ %
                         (("training_" if is_training else ""),
                          self.dataset_name, split, modality)]

    @contextmanager
    def _create_group(self, group_name):
        super().__setattr__('_group_name', group_name)
        yield
        super().__setattr__('_group_name', None)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if self._group_name:
            super().__getattribute__(self._group_name).__setattr__(key, value)

    def load_vocab_size(self):
        if os.path.exists(self.all_vocab_file):
            vocab = json.load(open(self.all_vocab_file, 'r'))
            self.size_vocab = len(vocab['vocab'])

    def load_info(self):
        if os.path.exists(self.info_file):
            self.info.update(json.load(open(self.info_file, 'r')))

    def update_info(self, item=None, file=None, keys=None):
        if file:
            item = json.load(open(file, 'r'))
        if keys:
            for key in keys:
                self.info[key] = item[key]
        else:
            self.info.update(item)
        json.dump(self.info, open(self.info_file, 'w'), indent=4)


def main():
    model_config = MovieQAConfig()
    print(json.dumps(model_config.tunable_parameter.__dict__, indent=4))
    h = hashlib.new('ripemd160')
    h.update(json.dumps(model_config.tunable_parameter.__dict__, indent=4).encode())
    print(h.hexdigest())
    print(model_config.num_training_train_examples)


if __name__ == '__main__':
    main()

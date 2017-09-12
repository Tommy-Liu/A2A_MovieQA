from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from glob import glob


class MovieQAConfig(object):
    """Wrapper class for all hyperparameters."""
    NPY_PATTERN_ = '*.npy'

    def __init__(self):
        """Sets the default hyperparameters."""
        # Directory of data
        self.data_dir = '../MovieQA_benchmark/story/video_clips'
        self.matidx_dir = '../MovieQA_benchmark/story/matidx'
        self.subt_dir = '../MovieQA_benchmark/story/subtt'
        self.video_img_dir = './video_img'
        self.feature_dir = './features'
        self.dataset_dir = './dataset'

        # File names
        self.avail_video_metadata_file = './avail_video_metadata.json'
        self.avail_video_subtitle_file = './avail_video_subtitle.json'
        self.avail_split_qa_file = './avail_split_qa.json'
        self.avail_tokenize_qa_file = './avail_tokenize_qa.json'
        self.avail_encode_qa_file = './encode_qa.json'
        self.sep_vocab_file = './avail_separate_vocab.json'
        self.all_vocab_file = './avail_all_vocab.json'
        self.info_file = './info.json'
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
        self.min_filter_size = 3
        self.max_filter_size = 5
        self.filter_sizes = list(range(self.min_filter_size,
                                       self.max_filter_size + 1))
        self.sliding_dim = 1024

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 512
        self.num_lstm_units = 512

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        self.size_vocab_q = 0
        self.size_vocab_a = 0
        self.size_vocab_s = 0

        self.grab_info()

        self.num_worker = 4
        # Optimizer for training the model.
        self.optimizer = "SGD"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        # Learning rate when fine tuning the Inception v3 parameters.
        self.train_inception_learning_rate = 0.0005

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5

    def grab_info(self):
        if not os.path.exists(self.info_file):
            avail_preprocessing_qa = json.load(open(self.sep_vocab_file, 'r'))
            self.size_vocab_q = len(avail_preprocessing_qa['vocab_q'])
            self.size_vocab_a = len(avail_preprocessing_qa['vocab_a'])
            self.size_vocab_s = len(avail_preprocessing_qa['vocab_s'])
            json.dump({
                'size_vocab_q': self.size_vocab_q,
                'size_vocab_a': self.size_vocab_a,
                'size_vocab_s': self.size_vocab_s,
            }, open(self.info_file, 'w'))
        else:
            info = json.load(open(self.info_file, 'r'))
            self.size_vocab_q = info['size_vocab_q']
            self.size_vocab_a = info['size_vocab_a']
            self.size_vocab_s = info['size_vocab_s']


def main():
    model_config = MovieQAConfig()
    print(model_config.size_vocab_q,
          model_config.size_vocab_a,
          model_config.size_vocab_s)


if __name__ == '__main__':
    main()

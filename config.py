from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob

import json
import os

json_file = './avail_preprocessing_qa.json'
info_file = './info.json'


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    NPY_PATTERN_ = '*.npy'
    feature_dir = './features'

    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None

        # Image format ("jpeg" or "png").
        self.image_format = "jpeg"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1
        self.num_worker = 4
        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/caption_ids"

        # Number of unique words in the vocab (plus 1, for <UNK>).
        # The default value is larger than the expected actual vocab size to allow
        # for differences between tokenizer versions used in preprocessing. There is
        # no harm in using a value greater than the actual vocab size, but using a
        # value less than the actual vocab size will result in an error.
        self.vocab_size = 12000
        self.feature_dim = 1536
        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4
        self.npy_files = glob(os.path.join(self.feature_dir, self.NPY_PATTERN_))
        # Batch size.
        self.batch_size = 4

        # File containing an Inception v3 checkpoint to initialize the variables
        # of the Inception model. Must be provided when starting training for the
        # first time.
        self.inception_checkpoint_file = None
        self.min_filter_size = 2
        self.max_filter_size = 5
        self.filter_sizes = list(range(self.min_filter_size,
                                       self.max_filter_size + 1))
        self.sliding_dim = 1024
        # Dimensions of Inception v3 input images.
        self.image_height = 299
        self.image_width = 299

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

    def grab_info(self):
        if not os.path.exists(info_file):
            avail_preprocessing_qa = json.load(open(json_file, 'r'))
            self.size_vocab_q = len(avail_preprocessing_qa['vocab_q'])
            self.size_vocab_a = len(avail_preprocessing_qa['vocab_a'])
            self.size_vocab_s = len(avail_preprocessing_qa['vocab_s'])
            json.dump({
                'size_vocab_q': self.size_vocab_q,
                'size_vocab_a': self.size_vocab_a,
                'size_vocab_s': self.size_vocab_s,
            }, open(info_file, 'w'))
        else:
            info = json.load(open(info_file, 'r'))
            self.size_vocab_q = info['size_vocab_q']
            self.size_vocab_a = info['size_vocab_a']
            self.size_vocab_s = info['size_vocab_s']


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 586363

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


def main():
    model_config = ModelConfig()
    print(model_config.size_vocab_q,
          model_config.size_vocab_a,
          model_config.size_vocab_s)


if __name__ == '__main__':
    main()

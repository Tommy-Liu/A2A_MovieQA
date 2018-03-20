import hashlib
import json
import os
from contextlib import contextmanager
from glob import glob
from os.path import join


class MovieQAParameter(object):
    def __init__(self):
        self.dropout_prob = dict(default=0.7, help='Probability of keeping value in dropout layer.', type=float)
        self.optimizer = dict(default="Adam", help='Optimization policy.', )

        # Learning rate for the initial phase of training.
        self.learning_rate = dict(default=0.0001, help='Initial learning rate.', type=float)
        self.decay_factor = dict(default=0.87, help='Decay factor of learning rate.', type=float)
        self.decay_epoch = dict(default=1.0, help='Number of epochs decay occurs.', type=float)
        self.batch_size = dict(default=2, help='Batch size. (no idea? Google it, idiot!)', type=int)
        self.clip_norm = dict(default=1.0, help='Norm value of gradient clipping.', type=float)
        self.initial_scale = dict(default=0.01, help='Initial value of weight\'s scale.', type=float)
        self.initializer = \
            dict(default='glorot', help='Initializer of weight.\n(truncated / random / orthogonal / glorot)')


class MovieQAPath(object):
    def __init__(self):
        self.benchmark_dir = '/mnt/data/tommy8054/MovieQA_benchmark'
        # Directory of original data
        self.story_dir = join(self.benchmark_dir, 'story')
        self.video_clips_dir = join(self.story_dir, 'video_clips')
        self.frame_time_dir = join(self.story_dir, 'matidx')
        self.subtitle_dir = join(self.story_dir, 'subtt')
        self.shot_boundary_dir = join(self.story_dir, 'shot_boundaries')

        # Directory of processed data
        self.data_dir = join(self.benchmark_dir, 'data')
        # Directory of all images of video clips
        self.image_dir = join(self.data_dir, 'images')
        # Directory of all features of video clips
        self.feature_dir = join(self.data_dir, 'features')
        # Directory of all subtitle sentence embedding
        self.encode_dir = join(self.data_dir, 'encode')
        # Directory of tfrecords
        self.dataset_dir = join(self.data_dir, 'dataset')

        # Experiment directory
        self.checkpoint_dir = './checkpoint'
        self.log_dir = './log'
        self.exp_dir = './exp'

        # Video data
        self.video_data_file = join(self.data_dir, 'video_data.json')
        # Images' file names
        self.images_name_file = join(self.data_dir, 'images_name.json')
        # Shot boundary
        self.shot_boundary_file = join(self.data_dir, 'shot_boundary.json')
        # Subtitle data TODO
        self.subtitle_file = join(self.data_dir, 'subtitle.json')
        # Time to frame TODO
        self.frame_time_file = join(self.data_dir, 'frame_time.json')
        # Subtitle shot boundary
        self.subtitle_shot_file = join(self.data_dir, 'video_subtitle_shot.json')

        # Vocabulary frequency file
        self.freq_file = join(self.data_dir, 'frequency.json')
        # Temporary subtitle file
        self.temp_subtitle_file = join(self.data_dir, 'temp_subt.json')

        # Tokenize files for sanity check
        self.tokenize_qa = join(self.data_dir, 'tokenize_qa.json')
        self.tokenize_subt = join(self.data_dir, 'tokenize_subt.json')

        # Encoded subtitle
        self.encode_subtitle_file = join(self.data_dir, 'encode_subtitle.json')
        # Encoded QA
        self.encode_qa_file = join(self.data_dir, 'encode_qa.json')
        # Vocabulary file
        self.vocab_file = join(self.data_dir, 'vocab.json')
        # Embedding file
        self.embedding_file = join(self.data_dir, 'embedding.npy')

        self.ques_file = join(self.data_dir, 'ques.npy')
        self.ans_file = join(self.data_dir, 'ans.npy')

        # Original qa TODO
        self.qa_file = join(self.data_dir, 'qa.json')
        # Original movie infomation
        self.movies_file = join(self.data_dir, 'movies.json')
        # Original qa split
        self.splits_file = join(self.data_dir, 'splits.json')
        # Npy file pattern
        self.npy_files = glob(os.path.join(self.feature_dir, '*.npy'))


class ExtendedObject(object):
    _group_name = None

    @contextmanager
    def _create_group(self, group_name):
        super().__setattr__('_group_name', group_name)
        yield
        super().__setattr__('_group_name', None)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if self._group_name:
            super().__getattribute__(self._group_name).__setattr__(key, value)


class Config(ExtendedObject):
    def __init__(self):
        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5

        self.tunable_parameter = ExtendedObject()
        with self._create_group('tunable_parameter'):
            # If < 1.0, the dropout keep probability applied to LSTM variables.
            self.lstm_dropout_keep_prob = 0.7

            # Optimizer for training the model.
            self.optimizer = "Adam"

            # Learning rate for the initial phase of training.
            self.initial_learning_rate = 0.0001
            self.learning_rate_decay_factor = 0.87
            self.num_epochs_per_decay = 1.0

            # Number of epochs
            self.num_epochs = 20


class MovieQAConfig(Config):
    """Wrapper class for all hyperparameters."""
    NPY_PATTERN_ = '*.npy'
    DATASET_PATTERN_ = '%s%s_%s_%s'  # 1. is_training 2. datasetname 3. split 4. modality
    TFRECORD_PATTERN_ = DATASET_PATTERN_ + '_%05d-of-%05d.tfrecord'
    TFRECORD_FILE_PATTERN_ = TFRECORD_PATTERN_.replace('%05d-of-', '*')
    NUMEXAMPLE_PATTERN_ = 'num_' + DATASET_PATTERN_ + '_examples'
    _group_name = None

    def __init__(self, level='.'):
        super(MovieQAConfig, self).__init__()
        # Directory of data
        self.directory = Config()
        with self._create_group('directory'):
            self.movieqa_benchmark_dir = '/mnt/data/tommy8054/MovieQA_benchmark'
            # Directory of original data
            self.video_clips_dir = join(self.movieqa_benchmark_dir, 'story/video_clips')
            self.matidx_dir = join(self.movieqa_benchmark_dir, 'story/matidx')
            self.subt_dir = join(self.movieqa_benchmark_dir, 'story/subtt')
            self.shot_boundary_dir = join(self.movieqa_benchmark_dir, 'story/shot_boundaries')

            # Directory of processed data
            self.data_dir = join(level, 'data')
            # Directory of all images of video clips
            self.video_img_dir = join(self.data_dir, 'video_img')
            # Directory of all features of video clips
            self.feature_dir = join(self.data_dir, 'features')
            # Directory of tfrecords
            self.dataset_dir = join(self.data_dir, 'dataset')
            # Directory of embedding things
            self.embedding_dir = './embedding'

            # Experiment directory
            self.checkpoint_dir = './checkpoint'
            self.log_dir = './log'
            self.exp_dir = './exp'

        # File names
        self.file_names = Config()
        with self._create_group('file_names'):
            # Video data
            self.video_data_file = join(self.data_dir, 'video_data.json')
            # Images' file names
            self.images_name_file = join(self.data_dir, 'images_name.json')
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
            self.glove_file = join(self.data_dir, 'glove.840B.300d.txt')

            # Word vocabulary of qa
            self.qa_word_vocab_file = join(self.embedding_dir, 'qa_word_vocab.json')

            # Counter of char
            self.qa_char_counter_file = join(self.embedding_dir, 'qa_char_counter.json')
            self.embed_char_counter_file = join(self.embedding_dir, 'embed_char_counter.json')

            # Char embedding
            self.char_vocab_file = join(self.embedding_dir, 'char_vocab.json')
            self.encode_embedding_key_file = join(self.embedding_dir, 'encode_embedding.npy')
            self.encode_embedding_len_file = join(self.embedding_dir, 'encode_embedding_len.npy')
            self.encode_embedding_vec_file = join(self.embedding_dir, 'encode_embedding_vec.npy')

            # Total embedding keys and values
            self.w2v_embedding_key_file = join(self.embedding_dir, 'w2v_embedding.json')
            self.w2v_embedding_vec_file = join(self.embedding_dir, 'w2v_embedding.npy')
            self.ft_embedding_key_file = join(self.embedding_dir, 'ft_embedding.json')
            self.ft_embedding_vec_file = join(self.embedding_dir, 'ft_embedding.npy')
            self.glove_embedding_key_file = join(self.embedding_dir, 'glove_embedding.json')
            self.glove_embedding_vec_file = join(self.embedding_dir, 'glove_embedding.npy')

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

        with self._create_group('tunable_parameter'):
            self.min_filter_size = 3
            self.max_filter_size = 5

            self.sliding_dim = 1024
            # LSTM input and output dimensionality, respectively.
            self.embedding_size = 512
            self.num_lstm_units = 512

            # Number of sliding convolution layer
            self.num_layers = 1

            # If not None, clip gradients to this value.
            self.clip_gradients = 1.0
            # Default modality
            self.modality = 'fixed_num'

        self.filter_sizes = list(range(self.min_filter_size,
                                       self.max_filter_size + 1))
        self.info = {}
        self.load_info()

    def get_num_example(self, split='train', modality='fixed_num',
                        is_training=False):
        return self.info[self.NUMEXAMPLE_PATTERN_ %
                         (("training_" if is_training else ""),
                          self.dataset_name, split, modality)]

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

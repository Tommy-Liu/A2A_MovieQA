import argparse
from os.path import join


class EmbeddingPath(object):
    def __init__(self, root=''):
        # Path
        self.data_dir = join(root, 'embed', 'data')
        self.log_dir = join(root, 'log')
        self.checkpoint_dir = join(root, 'checkpoint')

        # Word embedding, Vocabulary file
        self.fasttext_file = join(self.data_dir, 'crawl-300d-2M.vec')
        self.word2vec_file = join(self.data_dir, 'GoogleNews-vectors-negative300.txt')
        self.glove_file = join(self.data_dir, 'glove.840B.300d.txt')

        # Char embedding
        self.gram_counter_file = join(self.data_dir, 'gram_counter.json')
        self.gram_vocab_file = join(self.data_dir, 'gram_vocab.json')
        self.encode_embedding_key_file = join(self.data_dir, 'encode_embedding.npy')
        self.encode_embedding_len_file = join(self.data_dir, 'encode_embedding_len.npy')
        self.encode_embedding_vec_file = join(self.data_dir, 'encode_embedding_vec.npy')

        # All embedding keys and values
        self.w2v_embedding_key_file = join(self.data_dir, 'w2v_embedding.json')
        self.w2v_embedding_vec_file = join(self.data_dir, 'w2v_embedding.npy')
        self.ft_embedding_key_file = join(self.data_dir, 'ft_embedding.json')
        self.ft_embedding_vec_file = join(self.data_dir, 'ft_embedding.npy')
        self.glove_embedding_key_file = join(self.data_dir, 'glove_embedding.json')
        self.glove_embedding_vec_file = join(self.data_dir, 'glove_embedding.npy')

        # Trained embedding
        self.gram_embedding_vec_file = join(self.data_dir, 'gram_embedding.npy')
        # Parameters
        self.max_length = 17
        self.embedding_size = 300
        self.target = 'glove'


class EmbeddingParameter(object):
    def __init__(self):
        # ##################   Training Setting   #########################################################
        self.learning_rate = dict(default=1E-2, help='Initial learning rate.', type=float)
        self.batch_size = dict(default=128, help='Batch size of training.', type=int)
        self.epoch = dict(default=200, help='Training epochs', type=int)
        self.decay_epoch = dict(default=2, help='Span of epochs at decay.', type=int)
        self.decay_rate = dict(default=0.97, help='Decay rate.', type=float)
        self.optimizer = dict(default='sgd', help='Training policy (adam / momentum / sgd / rms).')
        self.clip_norm = dict(default=1.0, help='Norm value of gradient clipping.', type=float)
        # #################################################################################################
        # ##################   Initializer args   #########################################################
        self.initial_scale = dict(default=0.01, help='Initial value of weight\'s scale.', type=float)
        self.initializer = dict(default='glorot',
                                help='Initializer of weight.\n(truncated / random / orthogonal / glorot)')
        # #################################################################################################


# The code here should not be modified due to the consistency of the code and training setting.
def args_parse():
    hyper_parameters = vars(EmbeddingParameter())

    function_args = {
        'continue': dict(action='store_true', help='Continue the experiment.'),
        'auto': dict(action='store_true', help='Automatically choose a set of hyper-parameters.')}

    odds = {
        'checkpoint_file': dict(default=None, help='Checkpoint file'),
        'debug': dict(action='store_true', help='Debug mode.')}

    parser = argparse.ArgumentParser()
    for k in hyper_parameters:
        parser.add_argument('--' + k, **hyper_parameters[k])
    func_group = parser.add_mutually_exclusive_group()
    for k in function_args:
        func_group.add_argument('--' + k, **function_args[k])
    for k in odds:
        parser.add_argument('--' + k, **odds[k])

    args = parser.parse_args()
    args = vars(args)
    hp = {k: args[k] for k in hyper_parameters}
    rest = {k: args[k] for k in list(function_args.keys()) + list(odds.keys())}
    return hp, rest, parser


if __name__ == '__main__':
    args_parse()

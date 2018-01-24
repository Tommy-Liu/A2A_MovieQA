import argparse
from os.path import join


class CommonParameter(object):
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

        # Total embedding keys and values
        self.w2v_embedding_key_file = join(self.data_dir, 'w2v_embedding.json')
        self.w2v_embedding_vec_file = join(self.data_dir, 'w2v_embedding.npy')
        self.ft_embedding_key_file = join(self.data_dir, 'ft_embedding.json')
        self.ft_embedding_vec_file = join(self.data_dir, 'ft_embedding.npy')
        self.glove_embedding_key_file = join(self.data_dir, 'glove_embedding.json')
        self.glove_embedding_vec_file = join(self.data_dir, 'glove_embedding.npy')
        # Parameters
        self.max_length = 18
        self.embedding_size = 300
        self.target = 'glove'


# The code here should not be modified due to the consistency of the code and training setting.
def args_parse():
    hyper_parameters = {
        # ##################   Training Setting   ######################################################
        'learning_rate': {'default': 1E-2, 'help': 'Initial learning rate.', 'type': float},
        'batch_size': {'default': 128, 'help': 'Batch size of training.', 'type': int},
        'epoch': {'default': 200, 'help': 'Training epochs', 'type': int},
        'decay_epoch': {'default': 2, 'help': 'Span of epochs at decay.', 'type': int},
        'decay_rate': {'default': 0.97, 'help': 'Decay rate.', 'type': float},
        'optimizer': {'default': 'sgd', 'help': 'Training policy (adam / momentum / sgd / rms).',
                      'type': str},
        'clip_norm': {'default': 1.0, 'help': 'Norm value of gradient clipping.', 'type': float},
        # ##############################################################################################
        # ##################   Initializer args   ######################################################
        'initial_scale': {'default': 0.01, 'help': 'Initial value of weight\'s scale.', 'type': float},
        'initializer': {'default': 'glorot',
                        'help': 'Initializer of weight.\n(truncated / random / orthogonal / glorot)',
                        'type': str}
        # ###############################################################################################
    }

    function_args = {
        'continue': {'action': 'store_true', 'help': 'Continue the experiment.'},
        'auto': {'action': 'store_true', 'help': 'Automatically choose a set of hyper-parameters.'}}

    odds = {
        'checkpoint_file': {'default': None, 'help': 'Checkpoint file'},
        'debug': {'action': 'store_true', 'help': 'Debug mode.'}}

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

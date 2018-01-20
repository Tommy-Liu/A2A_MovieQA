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
    parser = argparse.ArgumentParser()
    # ##################   Training Setting   #########################################################################
    parser.add_argument('--continue', action='store_true', help='Continue the experiment.')
    parser.add_argument('--checkpoint_file', default=None, help='Checkpoint file')
    parser.add_argument('--learning_rate', default=1E-5, help='Initial learning rate.', type=float)
    parser.add_argument('--batch_size', default=32, help='Batch size of training.', type=int)
    parser.add_argument('--epoch', default=200, help='Training epochs', type=int)
    parser.add_argument('--decay_epoch', default=2, help='Span of epochs at decay.', type=int)
    parser.add_argument('--decay_rate', default=0.87, help='Decay rate.', type=float)
    parser.add_argument('--optimizer', default='sgd', help='Training policy (adam / momentum / sgd / rms).')
    parser.add_argument('--clip_norm', default=1.0, help='Norm value of gradient clipping.', type=float)
    # #################################################################################################################

    # ##################   Initializer args   ##########################################################################
    parser.add_argument('--initial_mean', default=0.0, help='Initial value of weight\'s mean.', type=float)
    parser.add_argument('--initial_scale', default=0.0075, help='Initial value of weight\'s scale.', type=float)
    parser.add_argument('--initializer', default='glorot',
                        help='Initializer of weight.\n(truncated / random / orthogonal / glorot)')
    # ##################################################################################################################

    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    args = parser.parse_args()

    return vars(args), parser


if __name__ == '__main__':
    # a, p = args_parse()
    # print(a)
    a = CommonParameter()
    print(a.fasttext_file)

import argparse


def args_parse():
    parser = argparse.ArgumentParser(add_help=False)
    # ##################   Program Mode   #############################################################################
    subparsers = parser.add_subparsers(dest='mode')
    parser_p = subparsers.add_parser('process', help='Process the data which creating tfrecords needs.')
    parser_i = subparsers.add_parser('inspect', help='Inspect the data stat.')
    parser_t = subparsers.add_parser('tfrecord', help='Create tfrecords.')
    # #################################################################################################################

    # ##################   Embedding Target   #########################################################################
    parser_p.add_argument('--target', default='glove', help='Learning target of word embedding.')
    parser_t.add_argument('--sorted', action='store_true', help='Divide data into groups of same length')
    parser_t.add_argument('--num_shards', default=128, help='Number of tfrecords.', type=int)
    parser_t.add_argument('--num_per_shard', default=10000, help='Number of instances in a shard.', type=int)
    parser.add_argument('--max_length', default=12, help='Maximal word length.', type=int)
    # #################################################################################################################

    # ##################   Training Setting   #########################################################################
    parser.add_argument('--reset', action='store_true', help='Reset the experiment.')
    parser.add_argument('--raw_input', action='store_true', help='Use raw data as input without tfreord.')
    parser.add_argument('--give_shards', default=1, help='Number of training shards given', type=int)
    parser.add_argument('--checkpoint_file', default=None, help='Checkpoint file')
    parser.add_argument('--learning_rate', default=1E-3, help='Initial learning rate.', type=float)
    parser.add_argument('--batch_size', default=32, help='Batch size of training.', type=int)
    parser.add_argument('--epoch', default=200, help='Training epochs', type=int)
    parser.add_argument('--decay_epoch', default=2, help='Span of epochs at decay.', type=int)
    parser.add_argument('--decay_rate', default=0.97, help='Decay rate.', type=float)
    parser.add_argument('--optimizer', default='adam', help='Training policy (adam / momentum / sgd / rms).')
    parser.add_argument('--loss', default='mse', help='Fist loss function. (mse / cos / abs / l2 / huber / mpse)')
    parser.add_argument('--sec_loss', default='mpse',
                        help='Second loss function. (mse / cos / abs / l2 / huber / mpse)')
    parser.add_argument('--clip_norm', default=1.0, help='Norm value of gradient clipping.', type=float)
    # #################################################################################################################

    # ##################   Model Setting   ############################################################################
    parser.add_argument('--dropout_prob', default=1.0, help='Probability of dropout.', type=float)
    parser.add_argument('--char_dim', default=64, help='Dimension of char embedding', type=int)
    parser.add_argument('--hidden_dim', default=256, help='Dimension of hidden state.', type=int)
    parser.add_argument('--conv_channel', default=512, help='Output channel of convolution layer.', type=int)
    parser.add_argument('--rnn', default='single', help='Multi / Single-layer rnn.')
    parser.add_argument('--nlayers', default=2, help='Number of Layers in rnn.', type=int)
    parser.add_argument('--rnn_cell', default='LSTM', help='RNN cell type. (GRU / LSTM / BasicRNN)')
    parser.add_argument('--model', default='matrice', help='Model modality.')
    # ##################################################################################################################

    # ##################   Initializer args   ##########################################################################
    parser.add_argument('--initializer', default='truncated',
                        help='Initializer of weight.\n(identity / truncated / random / orthogonal / glorot)')
    parser.add_argument('--bias_init', default=1.0, help='RNN cell bias initialization value.', type=float)
    parser.add_argument('--init_mean', default=0.0, help='Initial value of weight\'s mean.', type=float)
    parser.add_argument('--init_scale', default=0.075, help='Initial value of weight\'s scale.', type=float)
    # ##################################################################################################################

    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    args = parser.parse_args()

    return args, parser


if __name__ == '__main__':
    a, p = args_parse()
    print(a)
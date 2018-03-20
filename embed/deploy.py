from os.path import join

import numpy as np
import tensorflow as tf

import utils.data_utils as du
import utils.func_utils as fu
from embed.args import EmbeddingPath

_ep = EmbeddingPath()


def main():
    # data = EmbeddingData('embedding')
    # exp_paths = glob(join(cp.log_dir, 'embedding', '**', '*.json'), recursive=True)
    # min_exp = du.json_load(exp_paths[0])
    # min_idx = 0
    # for idx, p in enumerate(exp_paths):
    #     experiment = du.json_load(p)
    #     if min_exp['loss'] > experiment['loss']:
    #         min_exp = experiment
    #         min_idx = idx

    checkpoint = tf.train.latest_checkpoint(join(_ep.checkpoint_dir, 'embedding_v2'))
    print(checkpoint)
    vocab = du.json_load(_ep.gram_vocab_file)
    vocab_size = len(vocab)
    embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size - 1, 300],
                                       tf.float32, tf.glorot_normal_initializer(), trainable=True)
    zero_vec = tf.get_variable("zero_vec", [1, 300], tf.float32, tf.zeros_initializer(), trainable=False)
    gram_matrix = tf.concat([embedding_matrix, zero_vec], axis=0)
    saver = tf.train.Saver(tf.global_variables())
    print('Start deploying.')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, checkpoint)
        print('Restore done.')
        embedding_matrix = sess.run(model.gram_matrix)
        print('Matrix extraction done.')
        fu.safe_remove(_ep.gram_embedding_vec_file)
        np.save(_ep.gram_embedding_vec_file, embedding_matrix)
        print('Saving done.')


if __name__ == '__main__':
    main()

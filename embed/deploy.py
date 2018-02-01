from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf

import utils.data_utils as du
import utils.func_utils as fu
from embed.args import EmbeddingPath
from embed.model import NGramModel
from embed.train import EmbeddingData

cp = EmbeddingPath()


def main():
    data = EmbeddingData('embedding')
    exp_paths = glob(join(cp.log_dir, 'embedding', '**', '*.json'), recursive=True)
    min_exp = du.json_load(exp_paths[0])
    min_idx = 0
    for idx, p in enumerate(exp_paths):
        experiment = du.json_load(p)
        if min_exp['loss'] > experiment['loss']:
            min_exp = experiment
            min_idx = idx

    checkpoint = tf.train.latest_checkpoint(
        join(cp.checkpoint_dir, 'embedding', exp_paths[min_idx].split('/')[-2]))
    print(checkpoint)

    model = NGramModel(data, None)
    saver = tf.train.Saver(tf.global_variables())
    print('Start deploying.')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, checkpoint)
        print('Restore done.')
        embedding_matrix = sess.run(model.gram_matrix)
        print('Matrix extraction done.')
        fu.safe_remove(cp.gram_embedding_vec_file)
        np.save(cp.gram_embedding_vec_file, embedding_matrix)
        print('Saving done.')


if __name__ == '__main__':
    main()

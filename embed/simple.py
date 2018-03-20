import argparse
import math
from os.path import join

import numpy as np
import tensorflow as tf
from tqdm import trange

from embed.args import EmbeddingPath
from utils import data_utils as du
from utils import func_utils as fu
from utils import model_utils as mu

_ep = EmbeddingPath()


def main():
    name = 'embedding_v3'
    log_dir = join(_ep.log_dir, name)
    checkpoint_dir = join(_ep.checkpoint_dir, name)
    if args.reset:
        fu.safe_remove(log_dir)
        fu.safe_remove(join(checkpoint_dir))
    fu.make_dirs(log_dir)
    fu.make_dirs(join(checkpoint_dir, 'best'))

    encoded_key = np.load(_ep.encode_embedding_key_file)
    embedding = np.load(_ep.encode_embedding_vec_file)
    vocab = du.json_load(_ep.gram_vocab_file)

    batch_size = args.batch_size
    vocab_size = len(vocab)
    num_per_epoch = len(encoded_key)
    step_per_epoch = int(math.ceil(num_per_epoch / batch_size))

    # Input Data
    key_placeholder = tf.placeholder(encoded_key.dtype, encoded_key.shape)
    embedding_placeholder = tf.placeholder(embedding.dtype, embedding.shape)
    dataset = tf.data.Dataset.from_tensor_slices((key_placeholder, embedding_placeholder))
    dataset = dataset.repeat().shuffle(buffer_size=3000000).batch(batch_size).prefetch(4)
    iterator = dataset.make_initializable_iterator()
    word, vec = iterator.get_next()

    # Model
    embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size - 1, 300],
                                       tf.float32, tf.glorot_normal_initializer(), trainable=True)
    zero_vec = tf.get_variable("zero_vec", [1, 300], tf.float32, tf.zeros_initializer(), trainable=False)
    gram_matrix = tf.concat([embedding_matrix, zero_vec], axis=0)

    gram_embedding = tf.nn.embedding_lookup(gram_matrix, word)

    output = tf.reduce_sum(gram_embedding, axis=1)

    # Training Setting
    loss = mu.get_loss('cos', vec, output)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = mu.get_lr(args.lr_policy, args.learning_rate, global_step,
                              args.decay_epoch * step_per_epoch, args.decay_rate)
    optimizer = mu.get_opt(args.optimizer, learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    # gradients, variables = list(zip(*grads_and_vars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    saver = tf.train.Saver(tf.global_variables())
    best_saver = tf.train.Saver(tf.global_variables())
    summaries_op = tf.summary.merge(
        [tf.summary.scalar('loss', loss),
         tf.summary.scalar('learning_rate', learning_rate)])
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    checkpoint = args.checkpoint or tf.train.latest_checkpoint(checkpoint_dir)

    config = tf.ConfigProto(allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess, tf.summary.FileWriter(log_dir) as sw:
        # Initialize all variables
        init_op.run()
        if checkpoint:
            print('Restore from', checkpoint)
            saver.restore(sess, checkpoint)
        min_loss = 2 ** 32 - 1
        sess.run(iterator.initializer, feed_dict={key_placeholder: encoded_key, embedding_placeholder: embedding})
        step = tf.train.global_step(sess, global_step)
        try:
            while True:
                step = tf.train.global_step(sess, global_step)
                epoch = math.floor(step / step_per_epoch) + 1
                with trange(epoch * step_per_epoch - step) as pbar:
                    for _ in pbar:
                        ops = [loss, global_step, summaries_op, train_op]
                        l, step, summary, _ = sess.run(ops)
                        pbar.set_description('[%03d] Loss: %.3f' % (epoch, l))
                        sw.add_summary(summary, tf.train.global_step(sess, global_step))
                        if min_loss > l:
                            min_loss = l
                            if step % 10000 == 0 and step > 60000:
                                best_saver.save(sess, join(checkpoint_dir, 'best', name), step)
                    # saver.save(sess, join(checkpoint_dir, name), step)
        except KeyboardInterrupt:
            saver.save(sess, join(checkpoint_dir, name), step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, help='Model used to train.', type=int)
    parser.add_argument('--reset', action='store_true', help='Reset the experiment.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint file.')
    parser.add_argument('--learning_rate', default=0.5, help='Learning rate.', type=float)
    parser.add_argument('--decay_epoch', default=2.0, help='Decay epochs.', type=float)
    parser.add_argument('--decay_rate', default=0.88, help='Learning rate decay rate.', type=float)
    parser.add_argument('--optimizer', default='sgd', help='Optimizer.')
    parser.add_argument('--lr_policy', default='exp', help='Learning rate decay policy.')
    args = parser.parse_args()
    main()

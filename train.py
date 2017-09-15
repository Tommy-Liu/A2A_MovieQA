import tensorflow as tf
import tensorflow.contrib.layers as l

from config import MovieQAConfig
from get_dataset import MovieQAData
from model import VLLabMemoryModel


class TrainManager(object):
    def __init__(self):
        self.config = MovieQAConfig()
        self.data = MovieQAData(self.config, dummy=True)
        self.model = VLLabMemoryModel(self.data, self.config)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.val_model = VLLabMemoryModel(self.data, self.config)
        self.loss = tf.losses.sigmoid_cross_entropy(self.data.label,
                                                    self.model.logits)
        self.val_loss = tf.losses.sigmoid_cross_entropy(self.data.label,
                                                        self.val_model.logits)
        self.global_step = tf.train.get_or_create_global_step()

        self.learning_rate = tf.train.exponential_decay(self.config.initial_learning_rate,
                                                        self.global_step,
                                                        self.config.num_epochs_per_decay * self.config.num_training_train_examples,
                                                        self.config.learning_rate_decay_factor,
                                                        staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = l.optimize_loss(self.loss, self.global_step, self.config.initial_learning_rate)
        self.checkpoint_dir = ''
        self.log_dir = ''

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True, )
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(config=config) as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            for i in range(10):
                _, loss, step = sess.run([self.train_op, self.loss, self.global_step])
                print(loss)
            for i in range(10):
                loss, step = sess.run([self.val_loss, self.global_step])
                print(loss)
                # def hyper_parameter_spread(self):
                #     self.learning_rate_pool =
                # @property
                # def learning_rate(self):

    def eval(self, sess):
        pass

    def _get_checkpoint_dir(self):
        self.config.tunable_parameter


def main(_):
    trainer = TrainManager()
    trainer.train()


if __name__ == '__main__':
    tf.app.run()

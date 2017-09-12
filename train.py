import tensorflow as tf

from config import MovieQAConfig
from get_dataset import MovieQAData
from model import VLLabMemoryModel


class TrainManager(object):
    def __init__(self):
        self.config = MovieQAConfig()
        self.data = MovieQAData()
        self.model = VLLabMemoryModel(self.data)
        self.loss = tf.losses.sigmoid_cross_entropy(self.data.label,
                                                    self.model.logits)
        self.opt = tf.train.AdamOptimizer()

    def hyper_parameter_spread(self):
        pass


def main(_):
    trainer = TrainManager()


if __name__ == '__main__':
    tf.app.run()

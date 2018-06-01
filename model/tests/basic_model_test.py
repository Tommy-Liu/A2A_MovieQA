import unittest
from os.path import join

from config import MovieQAPath
from model.basic_model import BasicModel


class BasicModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model = BasicModel()
        self.model._hp.parse()
        self.mp = MovieQAPath()

    def test_log_dir(self):
        self.assertEqual(join(self.mp.log_dir, 'model.basic_model'), self.model._log_dir)

    def test_checkpoint_dir(self):
        self.assertEqual(join(self.mp.checkpoint_dir, 'model.basic_model'), self.model._checkpoint_dir)

    def test_attn_dir(self):
        self.assertEqual(join(self.mp.attn_dir, 'model.basic_model'), self.model._attn_dir)


if __name__ == '__main__':
    unittest.main()

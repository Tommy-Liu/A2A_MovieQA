import unittest

from model.basic_hp import BasicHP


class HPTestCase(unittest.TestCase):
    def setUp(self):
        self.hp = BasicHP()

    def test_exception(self):
        self.assertRaises(ValueError, self.hp.__repr__)

    def test_default(self):
        self.hp.parse()
        self.assertEqual(repr(self.hp), '', 'Default')

    def test_single_1(self):
        self.hp.parse(['--learning_rate', '0.01'])
        self.assertEqual(repr(self.hp), 'learning_rate:%G' % 0.01, 'Single arg 1')

    def test_single_2(self):
        self.hp.parse(['--loss', 'cos'])
        self.assertEqual(repr(self.hp), 'loss:%s' % 'cos', 'Single arg 2')

    def test_multi_1(self):
        self.hp.parse(['--loss', 'cos', '--decay_type', 'exp'])
        self.assertEqual(repr(self.hp), 'decay_type:exp,loss:cos', 'Multi arg 1')

    def test_multi_2(self):
        self.hp.parse(['--loss', 'cos', '--decay_type', 'exp', '--learning_rate', '0.01'])
        self.assertEqual(repr(self.hp), 'decay_type:exp,learning_rate:%G,loss:cos' % 0.01, 'Multi arg 1')


if __name__ == '__main__':
    unittest.main()

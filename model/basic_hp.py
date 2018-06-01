import numbers
from argparse import ArgumentParser
from functools import wraps


def parse_assert(func):
    @wraps(func)
    def wrapper(self, **kwds):
        if self._is_parsed:
            return func(self, **kwds)
        else:
            raise ValueError('Call parse first!')

    return wrapper


class BasicHP(object):
    def __init__(self):
        self._parser = ArgumentParser()
        self._parser.add_argument('--learning_rate', default=10 ** (-3), help='Learning rate.', type=float)
        self._parser.add_argument('--decay_rate', default=0.88, help='Decay rate of learning rate.', type=float)
        self._parser.add_argument('--reg', default=0.01, help='Rate of weight regularization.', type=float)
        self._parser.add_argument('--decay_epoch', default=128, help='Decay epoch.', type=int)
        self._parser.add_argument('--loss', default='sparse_softmax', help='Criteria.', type=str)
        self._parser.add_argument('--decay_type', default='linear_cos', help='Decay policy of learning rate', type=str)
        self._parser.add_argument('--opt', default='powersign-ld', help='Optimization policy.', type=str)
        self._is_parsed = False
        self._args = None
        self._default_args = None

    def parse(self, args=None):
        if args is None:
            args = []
        self._args, _ = self._parser.parse_known_args(args)
        self._args = vars(self._args)
        self._default_args = {k: self._parser.get_default(k) for k in self._args}
        self._is_parsed = True
        return self

    @parse_assert
    def __repr__(self):
        return self.__str__()

    @parse_assert
    def __str__(self):
        string = []
        for k in sorted(self._args.keys()):
            if self._default_args[k] != self._args[k]:
                if isinstance(self._args[k], numbers.Number):
                    string.append(k + ':%G' % self._args[k])
                else:
                    string.append(k + ':%s' % self._args[k])
        return ','.join(string)

    @parse_assert
    def __getitem__(self, item):
        return self._args[item]


if __name__ == '__main__':
    print(BasicHP().parse())

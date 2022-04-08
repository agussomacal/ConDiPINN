from collections import namedtuple
from functools import partial

Bounds = namedtuple("Bounds", "lower upper")


class NamedPartial:
    def __init__(self, func, *args, **kwargs):
        self.f = partial(func, *args, **kwargs)
        self.__name__ = func.__name__ + "_" + "_".join(list(map(str, args))) + "_".join(
            ["{}{}".format(k, v) for k, v in kwargs.items()])

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.__name__

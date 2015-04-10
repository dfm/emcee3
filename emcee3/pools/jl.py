# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["JoblibPool"]

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None


class JoblibPool(object):

    def __init__(self, *args, **kwargs):
        if Parallel is None:
            raise ImportError("joblib")
        self.args = args
        self.kwargs = kwargs

    def map(self, func, iterable):
        dfunc = delayed(func)
        return Parallel(*(self.args), **(self.kwargs))(
            dfunc(a) for a in iterable
        )

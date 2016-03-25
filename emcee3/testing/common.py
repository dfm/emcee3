# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import numpy as np
from tempfile import NamedTemporaryFile

from .. import backends
from ..model import Model

__all__ = ["NormalWalker", "MetadataWalker", "UniformWalker"]


class NormalWalker(Model):

    def __init__(self, ivar, width=np.inf):
        self.ivar = ivar
        self.width = width

    def compute_log_prior(self, state):
        p = state.__coords__
        state.__log_likelihood__ = 0.0
        if np.any(np.abs(p) > self.width):
            state.__log_likelihood__ = -np.inf
        return state

    def compute_log_likelihood(self, state):
        p = state.__coords__
        state.__log_likelihood__ = -0.5 * np.sum(p ** 2 * self.ivar)
        return state

    def compute_grad_log_likelihood(self, state):
        p = state.__coords__
        state.__grad_log_likelihood__ = -p * self.ivar
        return state


class MetadataWalker(NormalWalker):

    def compute_log_likelihood(self, state):
        p = state.__coords__
        state.mean = np.mean(p)
        state.median = np.median(p)
        return super(MetadataWalker, self).compute_log_likelihood(state)


class UniformWalker(Model):

    def compute_log_prior(self, state):
        p = state.__coords__
        state.__log_prior__ = 0.0 if np.all((-1 < p) * (p < 1)) else -np.inf
        return state

    def compute_log_likelihood(self, state):
        state.__log_likelihood__ = 0.0
        return state


class TempHDFBackend(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        f = NamedTemporaryFile("w", delete=False)
        f.close()
        self.filename = f.name
        return backends.HDFBackend(f.name, "test", **(self.kwargs))

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)

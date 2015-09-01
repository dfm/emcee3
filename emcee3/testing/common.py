# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NormalWalker", "MetadataWalker", "UniformWalker"]

import os
import numpy as np
from tempfile import NamedTemporaryFile

from .. import backends
from ..model import Model


class NormalWalker(Model):

    def __init__(self, ivar, width=np.inf):
        self.ivar = ivar
        self.width = width

    def get_lnprior(self, state, **kwargs):
        p = state.coords
        if np.any(np.abs(p) > self.width):
            return -np.inf
        return 0.0

    def get_lnlike(self, state, compute_grad=False, **kwargs):
        p = state.coords
        if compute_grad:
            state._grad_lnlike = -p*self.ivar
        return -0.5 * np.sum(p ** 2 * self.ivar)


class MetadataWalker(NormalWalker):

    def get_lnlike(self, state):
        p = state.coords
        state.mean = np.mean(p)
        state.median = np.median(p)
        return super(MetadataWalker, self).get_lnlike(state)


class UniformWalker(Model):

    def get_lnprior(self, state):
        p = state.coords
        return 0.0 if np.all((-1 < p) * (p < 1)) else -np.inf

    def get_lnlike(self, state):
        return 0.0


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

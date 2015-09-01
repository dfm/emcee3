# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Model"]

import numpy as np

from .state import State


class Model(object):

    def __init__(self, lnlikefn=None, lnpriorfn=None, args=tuple()):
        if lnpriorfn is None:
            lnpriorfn = _default_lnprior_function
        self.lnpriorfn = lnpriorfn
        self.lnlikefn = lnlikefn
        self.args = args

    def setup(self, state, **kwargs):
        pass

    def get_lnprior(self, state, **kwargs):
        return self.lnpriorfn(state.coords, *(self.args))

    def get_lnlike(self, state, **kwargs):
        return self.lnlikefn(state.coords, *(self.args))

    def get_state(self, coords, **kwargs):
        state = State(coords, -np.inf, -np.inf, False)
        self.setup(state, **kwargs)

        # Compute the prior.
        state.lnprior = self.get_lnprior(state, **kwargs)
        if not np.isfinite(state.lnprior):
            state.lnprior = -np.inf
            return state

        # Compute the likelihood.
        state.lnlike = self.get_lnlike(state, **kwargs)
        if not np.isfinite(state.lnlike):
            state.lnlike = -np.inf
        return state

    def __call__(self, coords, **kwargs):
        return self.get_state(coords, **kwargs)


def _default_lnprior_function(x, *args):
    return 0.0

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

    def setup(self, coords):
        pass

    def get_metadata(self, coords):
        return None

    def get_lnprior(self, coords):
        return self.lnpriorfn(coords, *(self.args))

    def get_lnlike(self, coords):
        return self.lnlikefn(coords, *(self.args))

    def get_state(self, coords):
        self.setup(coords)
        state = State(coords, -np.inf, -np.inf, False)

        # Compute the prior.
        lnprior = self.get_lnprior(coords)
        if not np.isfinite(lnprior):
            meta = self.get_metadata(coords)
            if meta is not None:
                state.metadata.update(meta)
            return state

        # Update the prior value.
        state.lnprior = lnprior

        # Compute the likelihood.
        lnlike = self.get_lnlike(coords)

        # Update the metadata.
        meta = self.get_metadata(coords)
        if meta is not None:
            state.metadata.update(meta)

        # Update the likelihood value.
        if np.isfinite(lnlike):
            state.lnlike = lnlike
        return state

    def __call__(self, coords):
        return self.get_state(coords)


def _default_lnprior_function(x, *args):
    return 0.0

# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["State"]

import numpy as np

from .compat import iteritems


class State(object):

    def __init__(self, coords, lnprior, lnlike, accepted=False, **metadata):
        self.coords = coords
        self.lnprior = lnprior
        self.lnlike = lnlike
        self.accepted = accepted
        for k, v in iteritems(metadata):
            setattr(self, k, v)

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    @property
    def grad_lnprob(self):
        some = False

        try:
            g = self._grad_lnlike
        except AttributeError:
            g = np.zeros_like(self.coords)
        else:
            some = True

        try:
            g += self._grad_lnprior
        except AttributeError:
            pass
        else:
            some = True

        if some:
            return g
        raise ValueError("your model must compute the gradients using "
                         "'_grad_lnlike' and/or '_grad_lnprior'")

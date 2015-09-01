# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["State"]

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

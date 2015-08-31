# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["State"]


class State(object):

    __slots__ = (
        "coords", "lnprior", "lnlike", "accepted", "metadata"
    )

    def __init__(self, coords, lnprior, lnlike, accepted=True, metadata=None):
        self.coords = coords
        self.lnprior = lnprior
        self.lnlike = lnlike
        self.accepted = accepted
        self.metadata = metadata

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

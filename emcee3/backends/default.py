# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DefaultBackend"]

import numpy as np


class DefaultBackend(object):

    def __init__(self):
        self._data = None
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.

        """
        self.niter = 0
        self.size = 0
        self.nwalkers = None
        self.spec = None
        del self._data
        self._data = None

    def check_dimensions(self, ens):
        if self.nwalkers is None:
            self.nwalkers = ens.nwalkers
        if self.spec is None:
            self.spec = ens.get_spec()
        if self.nwalkers != ens.nwalkers:
            raise ValueError("Dimension mismatch")

    def extend(self, n):
        k = self.nwalkers
        self.size = l = self.niter + n
        if self._data is None:
            self._data = dict(
                (name, np.empty((l, k) + shape, dtype=dtype))
                for name, shape, dtype in self.spec
            )
            self._acceptance = np.zeros(k, dtype=np.uint64)
        else:
            for name, shape, dtype in self.spec:
                self._data[name] = np.resize(
                    self._data[name],
                    (l, k) + shape
                )

    def update(self, ensemble):
        i = self.niter
        if i >= self.size:
            self.extend(i - self.size + 1)
        for name, _, _ in self.spec:
            ensemble.get_value(name, self._data[name][i])
        self._acceptance += ensemble.acceptance
        self.niter += 1

    def get_value(self, name):
        if self._data is None or name not in self._data:
            raise KeyError(name)
        return self._data[name][:self.niter]

    def __getattr__(self, name):
        try:
            return self.get_value(name)
        except KeyError:
            raise AttributeError(name)

    @property
    def coords(self):
        try:
            return self.get_value("coords")
        except KeyError:
            return None

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    @property
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / float(self.niter)

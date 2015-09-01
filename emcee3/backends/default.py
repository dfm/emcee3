# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DefaultBackend"]

import numpy as np


class DefaultBackend(object):

    def __init__(self):
        self._metadata = None
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.

        """
        # Clear the chain dimensions.
        self.niter = 0
        self.size = 0
        self.nwalkers, self.ndim = None, None

        # Clear the chain wrappers.
        self._coords = None
        self._lnprior = None
        self._lnlike = None
        self._metadata = None

    def check_dimensions(self, ens):
        if self.nwalkers is None:
            self.nwalkers = ens.nwalkers
        if self.ndim is None:
            self.ndim = ens.ndim
        if self.nwalkers != ens.nwalkers or self.ndim != ens.ndim:
            raise ValueError("Dimension mismatch")
        self.metadata_spec = ens.get_metadata_spec()

    def extend(self, n):
        k, d = self.nwalkers, self.ndim
        self.size = l = self.niter + n
        if self._coords is None:
            self._coords = np.empty((l, k, d), dtype=np.float64)
            self._lnprior = np.empty((l, k), dtype=np.float64)
            self._lnlike = np.empty((l, k), dtype=np.float64)
            self._acceptance = np.zeros(k, dtype=np.uint64)
            self._metadata = dict(
                (name, np.empty((l, k) + shape, dtype=dtype))
                for name, shape, dtype in self.metadata_spec
            )
        else:
            self._coords = np.resize(self._coords, (l, k, d))
            self._lnprior = np.resize(self._lnprior, (l, k))
            self._lnlike = np.resize(self._lnlike, (l, k))
            for name, shape, dtype in self.metadata_spec:
                self._metadata[name] = np.resize(
                    self._metadata[name],
                    (l, k) + shape
                )

    def update(self, ensemble):
        i = self.niter
        if i >= self.size:
            self.extend(i - self.size + 1)
        ensemble.get_coords(self._coords[i])
        ensemble.get_lnprior(self._lnprior[i])
        ensemble.get_lnlike(self._lnlike[i])
        for name, _, _ in self.metadata_spec:
            ensemble.get_metadata(name, self._metadata[name][i])
        self._acceptance += ensemble.acceptance
        self.niter += 1

    def get_metadata(self, name):
        if self._metadata is None or name not in self._metadata:
            raise KeyError(name)
        return self._metadata[name][:self.niter]

    @property
    def coords(self):
        return None if self._coords is None else self._coords[:self.niter]

    @property
    def lnprior(self):
        return self._lnprior[:self.niter]

    @property
    def lnlike(self):
        return self._lnlike[:self.niter]

    @property
    def lnprob(self):
        return self.lnprior + self.lnlike

    @property
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / float(self.niter)

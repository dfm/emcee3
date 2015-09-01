# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Ensemble"]

import numpy as np

from .pools import DefaultPool
from .compat import izip, iteritems


class Ensemble(object):
    """
    The state of the ensemble of walkers.

    :param model:
        The model specification. Should be, inherit from, or otherwise quack
        like a :class:`Model`.

    :param coords:
        The 2-D array of walker coordinate vectors. The shape of this array
        should be `(nwalkers, ndim)`.

    :param pool: (optional)
        A pool object that exposes a map function. This is especially useful
        for parallelization.

    :param random: (optional)
        A numpy-compatible random number generator. By default, this will be
        the built-in ``numpy.random`` module but if you want the ensemble to
        own its own state, you can supply an instance of
        ``numpy.random.RandomState``.

    .. note:: Any extra arguments or keyword arguments are pass along to the
              walker initialization.

    """
    def __init__(self, model, coords, *args, **kwargs):
        self.model = model
        self.pool = kwargs.pop("pool", DefaultPool())
        self.random = kwargs.pop("random", np.random.RandomState())

        # Interpret the dimensions of the ensemble.
        self._coords = np.atleast_1d(coords).astype(np.float64)
        if not len(self._coords.shape) == 2:
            raise ValueError("Invalid ensemble coordinate dimensions")
        self.nwalkers, self.ndim = self._coords.shape

        # Initialize the walkers at these coordinates.
        self.walkers = self.propose(self._coords)
        self.acceptance = np.ones(self.nwalkers, dtype=bool)

        if not np.all(np.isfinite(self.lnprob)):
            raise ValueError("invalid (zero-probability) coordinates")

    def propose(self, coords):
        """
        Given a new set of coordinates return arrays of log-prior and
        log-likelihood values.

        :param coords:
            The new coordinate matrix. It should be ``(nwalkers, ndim)``.

        """
        return list(self.pool.map(self.model, coords))

    def update(self, walkers, slice=slice(None)):
        """
        Update the coordinate matrix and probability containers given the
        current list of walkers. Moves should call this after proposing and
        accepting the walkers.

        """
        for j, s in izip(np.arange(self.nwalkers)[slice], walkers):
            self.acceptance[j] = s.accepted
            if s.accepted:
                self.walkers[j] = s
                if not np.isfinite(s.lnprob):
                    raise RuntimeError("invalid (zero-probability) proposal "
                                       "accepted")

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d.pop("pool", None)
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self.pool = DefaultPool()

    def __len__(self):
        return self.nwalkers

    def get_spec(self):
        md = self.walkers[0].__dict__
        return [(k, m.shape, np.asarray(m).dtype)
                for k, m in
                ((k0, np.asarray(m0)) for k0, m0 in iteritems(md)
                 if k0 is not "accepted")]

    def get_value(self, key, out=None):
        if out is None:
            v = np.asarray(getattr(self.walkers[0], key))
            out = np.empty((self.nwalkers, ) + v.shape, dtype=v.dtype)
        for i, s in enumerate(self.walkers):
            out[i] = getattr(s, key)
        return out

    def __getattr__(self, key):
        return self.get_value(key)

    def __getitem__(self, key):
        try:
            return self.get_value(key)
        except AttributeError:
            raise KeyError(key)

    # def get_metadata(self, key, out=None):
    #     if out is None:
    #         v = self.walkers[0].metadata[key]
    #         out = np.empty((self.nwalkers, ) + v.shape, dtype=v.dtype)
    #     for i, s in enumerate(self.walkers):
    #         out[i] = s.metadata[key]
    #     return out

    # def _get_value(self, name, shape, out):
    #     if out is None:
    #         out = np.empty(shape, dtype=np.float64)
    #     for i, s in enumerate(self.walkers):
    #         out[i] = getattr(s, name)
    #     return out

    # def get_coords(self, out=None):
    #     return self._get_value("coords", (self.nwalkers, self.ndim), out)

    # def get_lnprior(self, out=None):
    #     return self._get_value("lnprior", self.nwalkers, out)

    # def get_lnlike(self, out=None):
    #     return self._get_value("lnlike", self.nwalkers, out)

    # @property
    # def coords(self):
    #     """The coordinate vectors of the walkers."""
    #     return self.get_coords()

    # @property
    # def lnprior(self):
    #     """The ln-priors of the walkers up to a constant."""
    #     return self.get_lnprior()

    # @property
    # def lnlike(self):
    #     """The ln-likelihoods of the walkers up to a constant."""
    #     return self.get_lnlike()

    # @property
    # def lnprob(self):
    #     """The ln-probabilities of the walker up to a constant."""
    #     return self.lnprior + self.lnlike

# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .pools import DefaultPool
from .model import is_model, SimpleModel

__all__ = ["Ensemble"]


class Ensemble(object):
    """The state of the ensemble of walkers.

    Args:
        model (callable or Model): The model specification. This can be a
            callable, an instance of :class:`SimpleModel`, or another instance
            that quacks like a :class:`Model`. If ``model`` is a callable, it
            should take a coordinate vector as input and return the evaluated
            log-probability up to an additive constant.
        coords (Optional[array[nwalkers, ndim]]): The array of walker
            coordinate vectors.
        pool (Optional): A pool object that exposes a map function for
            parallelization purposes.
        random (Optional): A numpy-compatible random number generator. By
            default, this will be the built-in ``numpy.random`` module but if
            you want the ensemble to own its own state, you can supply an
            instance of ``numpy.random.RandomState``.

    """
    def __init__(self, model, coords, pool=None, random=None):
        if is_model(model):
            self.model = model
        else:
            if not callable(model):
                raise ValueError("the 'model' must have a "
                                 "'compute_log_probability' method or be "
                                 "callable")
            self.model = SimpleModel(model)
        self.pool = DefaultPool() if pool is None else pool
        self.random = np.random.RandomState() if random is None else random

        # Interpret the dimensions of the ensemble.
        self._coords = np.atleast_1d(coords).astype(np.float64)
        if not len(self._coords.shape) == 2:
            raise ValueError("Invalid ensemble coordinate dimensions")
        self.nwalkers, self.ndim = self._coords.shape

        # Initialize the walkers at these coordinates.
        self.walkers = self.propose(self._coords)
        self.acceptance = np.ones(self.nwalkers, dtype=bool)

        if not np.all(np.isfinite(self.log_probability)):
            raise ValueError("invalid or zero-probability coordinates")

    def propose(self, coords):
        """Given a new set of coordinates return arrays of log-prior and
        log-likelihood values.

        Args:
            coords (array[nwalkers, ndim]): The new coordinate matrix.

        Returns:
            list: A list of walker :class:`State` objects evaluated at the
                specified coordinates.

        """
        return list(self.pool.map(self.model, coords))

    def update(self, walkers, subset=None):
        """Update the coordinate matrix and probability containers given the
        current list of walkers. Moves should call this after proposing and
        accepting the walkers.

        Note:
            Only the walkers with ``accepted == True`` are updated.

        Args:
            walkers (list[State]): A list of walkers states.
            subset: If provided, ``walkers`` only corresponds to the indicated
                subset of the walkers.

        Raises:
            RuntimeError: If an invalid state is accepted.

        """
        if subset is None:
            subset = slice(None)

        for j, s in zip(np.arange(self.nwalkers)[subset], walkers):
            self.acceptance[j] = s.accepted
            if s.accepted:
                self.walkers[j] = s
                if not np.isfinite(s.log_probability):
                    raise RuntimeError("invalid or zero-probability proposal "
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

    @property
    def dtype(self):
        return self.walkers[0].dtype

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.walkers[key]
        try:
            return self.get_value(key)
        except (AttributeError, TypeError):
            return self.walkers[key]

    def get_value(self, key, out=None):
        if out is None:
            v = np.asarray(getattr(self.walkers[0], key))
            out = np.empty((self.nwalkers, ) + v.shape, dtype=v.dtype)
        for i, s in enumerate(self.walkers):
            out[i] = getattr(s, key)
        return out

    def __getattr__(self, key):
        return self.get_value(key)

# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from functools import wraps

__all__ = ["Backend"]


def _check_run(f):
    @wraps(f)
    def func(self, *args, **kwargs):
        if self.niter <= 0:
            raise AttributeError("You need to run the chain first or store "
                                 "the chain using the 'store' keyword "
                                 "argument to Sampler.sample")
        return f(self, *args, **kwargs)
    return func


class Backend(object):
    """The default backend that stores the data in memory as numpy arrays.

    The backend can be subscripted to access the data.

    Attributes:
        acceptance: An array of ``nwalkers`` integer acceptance counts.
        acceptance_fraction: An array of ``nwalkers`` acceptance fractions.

    """

    def __init__(self):
        self._data = None
        self.reset()

    def __len__(self):
        return self.niter

    def reset(self):
        """Clear the chain and reset it to its default state.

        """
        self.niter = 0
        self.size = 0
        self.nwalkers = None
        self.dtype = None
        del self._data
        self._data = None

    def check_dimensions(self, ensemble):
        """Check that an ensemble is consistent with the current chain.

        Args:
            ensemble (Ensemble): The ensemble to check.

        Raises:
            ValueError: If the dimension or data type of the ensemble is
                inconsistent with the stored data.

        """
        if self.nwalkers is None:
            self.nwalkers = ensemble.nwalkers
        if self.dtype is None:
            self.dtype = ensemble.dtype
        if self.nwalkers != ensemble.nwalkers:
            raise ValueError("Dimension mismatch")
        if self.dtype != ensemble.dtype:
            raise ValueError("Data type mismatch")

    def extend(self, n):
        """Extend the chain by a given number of steps.

        Args:
            n (int): The number of steps to extend the chain by.

        """
        k = self.nwalkers
        self.size = l = self.niter + n
        if self._data is None:
            self._data = np.empty((l, k), dtype=self.dtype)
            self._acceptance = np.zeros(k, dtype=np.uint64)
        else:
            dl = l - self._data.shape[0]
            if dl > 0:
                self._data = np.concatenate((
                    self._data, np.empty((dl, k), dtype=self._data.dtype)
                ), axis=0)

    def update(self, ensemble):
        """Append an ensemble to the chain.

        Args:
            ensemble (Ensemble): The ensemble to append.

        """
        i = self.niter
        if i >= self.size:
            self.extend(i - self.size + 1)
        for j, walker in enumerate(ensemble):
            self._data[i, j] = walker.to_array()
        self._acceptance += ensemble.acceptance
        self.niter += 1

    def __getitem__(self, name_and_index_or_slice):
        try:
            name, index_or_slice = name_and_index_or_slice
        except ValueError:
            name = name_and_index_or_slice
            index_or_slice = slice(None)
        return self._data[name][:self.niter][index_or_slice]

    @property
    @_check_run
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / float(self.niter)

    @property
    @_check_run
    def coords(self):
        return self.get_coords()

    @property
    @_check_run
    def log_prior(self):
        return self.get_log_prior()

    @property
    @_check_run
    def log_likelihood(self):
        return self.get_log_likelihood()

    @property
    @_check_run
    def log_probability(self):
        return self.get_log_probability()

    def get_coords(self, **kwargs):
        """
        Get the stored chain of MCMC samples. This will fail if no backend was
        used or if the chain wasn't stored.

        :param flat: (optional)
            Flatten the chain across the ensemble. (default: ``False``)

        :param thin: (optional)
            Take only every ``thin`` steps from the chain. (default: ``1``)

        :param discard: (optional)
            Discard the first ``discard`` steps in the chain as burn-in.
            (default: ``0``)

        """
        return self.get_value("coords", **kwargs)

    def get_log_prior(self, **kwargs):
        """
        Get the stored chain ln-prior values. This will fail if no backend was
        used or if the chain wasn't stored.

        :param flat: (optional)
            Flatten the chain across the ensemble. (default: ``False``)

        :param thin: (optional)
            Take only every ``thin`` steps from the chain. (default: ``1``)

        :param discard: (optional)
            Discard the first ``discard`` steps in the chain as burn-in.
            (default: ``0``)

        """
        return self.get_value("log_prior", **kwargs)

    def get_log_likelihood(self, **kwargs):
        """
        Get the stored chain ln-likelihood values. This will fail if no
        backend was used or if the chain wasn't stored.

        :param flat: (optional)
            Flatten the chain across the ensemble. (default: ``False``)

        :param thin: (optional)
            Take only every ``thin`` steps from the chain. (default: ``1``)

        :param discard: (optional)
            Discard the first ``discard`` steps in the chain as burn-in.
            (default: ``0``)

        """
        return self.get_value("log_likelihood", **kwargs)

    def get_log_probability(self, **kwargs):
        """
        Get the stored chain ln-probability values. This will fail if no
        backend was used or if the chain wasn't stored.

        :param flat: (optional)
            Flatten the chain across the ensemble. (default: ``False``)

        :param thin: (optional)
            Take only every ``thin`` steps from the chain. (default: ``1``)

        :param discard: (optional)
            Discard the first ``discard`` steps in the chain as burn-in.
            (default: ``0``)

        """
        return (
            self.get_value("log_prior", **kwargs) +
            self.get_value("log_likelihood", **kwargs)
        )

    def get_value(self, name, flat=False, thin=1, discard=0):
        v = self[name, discard::thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

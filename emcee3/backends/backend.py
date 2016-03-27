# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np

__all__ = ["Backend"]


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
            self._data = np.resize(self._data, (l, k))

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

    def __getitem__(self, index_or_slice):
        return self._data[:self.niter][index_or_slice]

    @property
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / float(self.niter)

    # def get_value(self, name):
    #     if self._data is None or name not in self._data.dtype.names:
    #         raise KeyError(name)
    #     return self._data[name, :self.niter]

    # def __getattr__(self, name):
    #     try:
    #         return self.get_value(name)
    #     except KeyError:
    #         raise AttributeError(name)

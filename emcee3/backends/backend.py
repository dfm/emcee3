# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np

from ..autocorr import integrated_time

__all__ = ["Backend"]


class Backend(object):
    """The default backend that stores the data in memory as numpy arrays.

    The backend can be subscripted to access the data.

    Attributes:
        acceptance: An array of ``nwalkers`` integer acceptance counts.
        acceptance_fraction: An array of ``nwalkers`` acceptance fractions.
        coords: An array of ``(niter, nwalkers, ndim)`` coordinates.
        log_prior: An array of ``(niter, nwalkers)`` log prior evaluations.
        log_likelihood: An array of ``(niter, nwalkers)`` log likelihood
            evaluations.
        log_probability: An array of ``(niter, nwalkers)`` log probability
            evaluations.

    """

    def __init__(self):
        self._data = None
        self.reset()

    def __len__(self):
        return self.niter

    def reset(self):
        """Clear the chain and reset it to its default state."""
        self.niter = 0
        self.size = 0
        self.nwalkers = None
        self.dtype = None
        del self._data
        self._data = None
        self._random_state = None

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
        self._random_state = ensemble.random.get_state()
        self.niter += 1

    def __getitem__(self, name_and_index_or_slice):
        if self.niter <= 0:
            raise AttributeError("You need to run the chain first or store "
                                 "the chain using the 'store' keyword "
                                 "argument to Sampler.sample")
        try:
            name, index_or_slice = name_and_index_or_slice
        except ValueError:
            name = name_and_index_or_slice
            index_or_slice = slice(None)
        return self._data[name][:self.niter][index_or_slice]

    def get_coords(self, **kwargs):
        """Get the stored chain of MCMC samples.

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        return self.get_value("coords", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the chain of log priors evaluated at the MCMC samples.

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log priors.

        """
        return self.get_value("log_prior", **kwargs)

    def get_log_likelihood(self, **kwargs):
        """Get the chain of log likelihoods evaluated at the MCMC samples.

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log likelihoods.

        """
        return self.get_value("log_likelihood", **kwargs)

    def get_log_probability(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples.

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        return (
            self.get_value("log_prior", **kwargs) +
            self.get_value("log_likelihood", **kwargs)
        )

    def get_integrated_autocorr_time(self, **kwargs):
        """Get the integrated autocorrelation time for each dimension.

        Any arguments are passed directly to :func:`autocorr.integrated_time`.

        Returns:
            array[ndim]: The estimated autocorrelation time in each dimension.

        """
        return integrated_time(np.mean(self.get_value("coords"), axis=1),
                               **kwargs)

    def get_value(self, name, flat=False, thin=1, discard=0):
        v = self[name, discard::thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    @property
    def acceptance(self):
        return self._acceptance

    @property
    def acceptance_fraction(self):
        return self.acceptance / float(self.niter)

    @property
    def coords(self):
        return self.get_coords()

    @property
    def log_prior(self):
        return self.get_log_prior()

    @property
    def log_likelihood(self):
        return self.get_log_likelihood()

    @property
    def log_probability(self):
        return self.get_log_probability()

    @property
    def random_state(self):
        return self._random_state

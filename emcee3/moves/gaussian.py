# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .mh import MHMove

__all__ = ["GaussianMove", "HeavyTailMove"]


class GaussianMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    Args:
        cov: The covariance of the proposal function. This can be a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general respectively.

    """
    def __init__(self, cov, factor=None, mode="vector"):
        # Parse the proposal type.
        try:
            float(cov)

        except TypeError:
            cov = np.atleast_1d(cov)
            if len(cov.shape) == 1:
                # A diagonal proposal was given.
                ndim = len(cov)
                proposal = _diagonal_proposal(np.sqrt(cov), factor, mode)

            elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                # The full, square covariance matrix was given.
                ndim = cov.shape[0]
                proposal = _proposal(cov, factor, mode)

            else:
                raise ValueError("Invalid proposal scale dimensions")

        else:
            # This was a scalar proposal.
            ndim = None
            proposal = _isotropic_proposal(np.sqrt(cov), factor, mode)

        super(GaussianMove, self).__init__(proposal, ndim=ndim)


class HeavyTailMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    Args:
        cov: The covariance of the proposal function. This can be a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general respectively.

    """
    def __init__(self, cov):
        pass


class _isotropic_proposal(object):

    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale
        if factor is None:
            self._log_factor = None
        else:
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(("'{0}' is not a recognized mode. "
                              "Please select from: {1}")
                             .format(mode, self.allowed_modes))
        self.mode = mode

    @property
    def factor(self):
        if self._log_factor is None:
            return 1.0
        return np.exp(np.random.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0):
        return x0 + self.factor * self.scale * rng.randn(*(x0.shape))

    def __call__(self, rng, x0):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _diagonal_proposal(_isotropic_proposal):

    def get_updated_vector(self, rng, x0):
        return x0 + self.factor * self.scale * rng.randn(*(x0.shape))


class _proposal(_isotropic_proposal):

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return x0 + self.factor * rng.multivariate_normal(
            np.zeros(len(self.scale)), self.scale)

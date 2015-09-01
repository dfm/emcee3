# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HMCMove"]

import numpy as np

from ..compat import izip


class _hmc_wrapper(object):

    def __init__(self, model, nsteps, epsilon, random):
        self.model = model
        self.nsteps = nsteps
        self.epsilon = epsilon
        self.random = random

    def leapfrog(self, theta, r):
        state = self.model.get_state(theta, compute_grad=True)
        r = r + 0.5 * self.epsilon * state._grad_lnprob
        theta = theta + self.epsilon * r
        state = self.model.get_state(theta, compute_grad=True)
        r += 0.5 * self.epsilon * state._grad_lnprob
        return theta, r, state

    def __call__(self, theta0):
        # Choose the number of steps to take.
        try:
            L = int(self.nsteps)
        except TypeError:
            L = int(self.random.uniform(self.nsteps[0], self.nsteps[1]))

        # Integrate the coordinates.
        r = r0 = np.random.randn(len(theta0))
        theta = theta0
        for i in range(L):
            theta, r, new_state = self.leapfrog(theta, r)

        return new_state, 0.5 * (np.dot(r0, r0) - np.dot(r, r))


class HMCMove(object):
    """

    """
    def __init__(self, nsteps, epsilon):
        self.nsteps = nsteps
        self.epsilon = epsilon

    def update(self, ensemble):
        """
        Execute a single step starting from the given :class:`Ensemble` and
        updating it in-place.

        :param ensemble:
            The starting :class:`Ensemble`.

        :return ensemble:
            The same ensemble updated in-place.

        """
        integrator = _hmc_wrapper(ensemble.model, self.nsteps, self.epsilon,
                                  ensemble.random)
        results = ensemble.pool.map(integrator, ensemble.coords)

        # Loop over the walkers and update them accordingly.
        states = []
        for (state, factor), walker in izip(results, ensemble.walkers):
            lnpdiff = state.lnprob - walker.lnprob + factor
            if lnpdiff > 0.0 or ensemble.random.rand() < np.exp(lnpdiff):
                state.accepted = True
            states.append(state)

        # Update the ensemble's coordinates and log-probabilities.
        ensemble.update(states)
        return ensemble

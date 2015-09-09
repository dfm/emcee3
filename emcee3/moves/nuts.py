# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NUTSMove"]

import numpy as np

from ..compat import izip, imap
from .hmc import _hmc_wrapper


class _nuts_wrapper(_hmc_wrapper):

    def __init__(self, model, random):
        self.model = model
        self.random = random

    def initial_epsilon_heuristic(self, theta):
        r = np.random.randn(len(theta))

        state0 = self.model.get_state(theta, compute_grad=False)
        lnp0 = state0.lnprob - 0.5 * np.dot(r, r)

        self.epsilon = 1.0
        _, rp, state = self.leapfrog(theta, r)
        lnp = state.lnprob - 0.5 * np.dot(rp, rp)
        a = 2.0 * (lnp - lnp0 > np.log(0.5)) - 1.0

        while a * (lnp - lnp0) > -a * np.log(2):
            self.epsilon *= 2. ** a
            _, rp, state = self.leapfrog(theta, r)
            lnp = state.lnprob - 0.5 * np.dot(rp, rp)
        return self.epsilon

    def build_tree(self, lnprob0, theta, r, lnu, v, j, delta_max=1000):
        # Root node case.
        if j == 0:
            lnprob0 -= 0.5 * np.dot(r, r)

            self.epsilon *= v
            theta, r, state = self.leapfrog(theta, r)
            lnp0 = state.lnprob
            self.epsilon *= v

            lnp = lnp0 - 0.5 * np.dot(r, r)
            n = float(lnu <= lnp)
            s = lnu < delta_max + lnp
            lnalpha = min(0.0, lnp - lnprob0)
            return (
                theta, r, lnp0, theta, r, lnp0, state, n, s, lnalpha, 1.0
            )

        # Recursively build the tree.
        theta_m, r_m, lnp_m, theta_p, r_p, lnp_p, state, n, s, lna, na = \
            self.build_tree(lnprob0, theta, r, lnu, v, j-1)
        if not s:
            return (
                theta_m, r_m, lnp_m, theta_p, r_p, lnp_m, state, n, s, lna, na
            )

        if v < 0:
            theta_m, r_m, lnp_m, _, _, _, state_pr, n_pr, s_pr, lna_pr, na_pr = \
                self.build_tree(lnp_m, theta_m, r_m, lnu, v, j-1)
        else:
            _, _, _, theta_p, r_p, lnp_p, state_pr, n_pr, s_pr, lna_pr, na_pr = \
                self.build_tree(lnp_p, theta_p, r_p, lnu, v, j-1)
        if n_pr and np.random.rand() < n_pr / (n + n_pr):
            state = state_pr
        delta_theta = theta_p - theta_m
        s = s_pr * (np.dot(delta_theta, r_m) >= 0.0) \
            * (np.dot(delta_theta, r_p) >= 0.0)
        n += n_pr
        lna = np.logaddexp(lna, lna_pr)
        na += na_pr

        return (
            theta_m, r_m, lnp_m, theta_p, r_p, lnp_p, state, n, s, lna, na
        )

    def __call__(self, state):
        theta = state.coords
        lnp = state.lnprob
        r = np.random.randn(len(theta))
        lnu = np.log(np.random.uniform(0, np.exp(lnp - 0.5*np.dot(r, r))))

        theta_m = theta_p = theta
        r_m = r_p = r
        lnp_m = lnp_p = lnp
        j = 0
        s = True
        n = 1.0

        while s:
            v = 1.0 - 2.0 * (np.random.rand() < 0.5)
            if v < 0:
                theta_m, r_m, lnp_m, _, _, _, state_pr, n_pr, s_pr, lna, na = \
                    self.build_tree(lnp_m, theta_m, r_m, lnu, v, j)
            else:
                _, _, _, theta_p, r_p, lnp_p, state_pr, n_pr, s_pr, lna, na = \
                    self.build_tree(lnp_p, theta_p, r_p, lnu, v, j)
            if s_pr and np.random.rand() <= n_pr / n:
                state = state_pr

            delta_theta = theta_p - theta_m
            s = s_pr * (np.dot(delta_theta, r_m) >= 0.0) \
                * (np.dot(delta_theta, r_p) >= 0.0)
            n += n_pr
            j += 1

        state._nuts_ln_alpha = lna
        state._nuts_num_a = na
        state._nuts_epsilon = self.epsilon
        state.accepted = True
        return state


class NUTSMove(object):
    """

    """
    def __init__(self, epsilon=None):
        self.base_epsilons = epsilon

    def _initialize_epsilon(self, integrator, walkers):
        self.base_epsilons = np.array([
            integrator.initial_epsilon_heuristic(w.coords)
            for w in walkers
        ])

    def update(self, ens):
        """
        Execute a single step starting from the given :class:`Ensemble` and
        updating it in-place.

        :param ensemble:
            The starting :class:`Ensemble`.

        :return ensemble:
            The same ensemble updated in-place.

        """
        integrator = _nuts_wrapper(ens.model, ens.random)
        if self.base_epsilons is None:
            self._initialize_epsilon(integrator, ens.walkers)
        integrator.epsilon = np.mean(self.base_epsilons)

        ens.update(list(ens.pool.map(integrator, ens.walkers)))
        return ens

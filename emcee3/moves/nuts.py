# -*- coding: utf-8 -*-

from __future__ import division, print_function

import copy
import numpy as np
from ..state import State
from .hmc import HamiltonianMove, _hmc_wrapper

__all__ = ["NoUTurnsMove"]


class _nuts_wrapper(_hmc_wrapper):

    def __init__(self, random, model, cov, epsilon, max_depth=500,
                 delta_max=1000.0):
        self.max_depth = max_depth
        self.delta_max = delta_max
        super(_nuts_wrapper, self).__init__(random, model, cov, epsilon)

    def leapfrog(self, state, epsilon):
        p = state._momentum + 0.5 * epsilon * state.grad_log_probability
        q = state.coords + epsilon * self.cov.apply(p)
        state = self.model.compute_log_probability(State(q))
        state = self.model.compute_grad_log_probability(state)
        state._momentum = p + 0.5 * epsilon * state.grad_log_probability
        return state

    def build_tree(self, state, u, v, j):
        if j == 0:
            state_pr = self.leapfrog(state, v * self.epsilon)
            K_pr = np.dot(state_pr._momentum,
                          self.cov.apply(state_pr._momentum))
            log_prob_pr = state_pr.log_probability - 0.5 * K_pr
            n_pr = int(np.log(u) < log_prob_pr)
            s_pr = np.log(u) - self.delta_max < log_prob_pr
            return state_pr, state_pr, state_pr, n_pr, s_pr

        # Recurse.
        state_m, state_p, state_pr, n_pr, s_pr = \
            self.build_tree(state, u, v, j - 1)
        if s_pr:
            if v < 0.0:
                state_m, _, state_pr_2, n_pr_2, s_pr_2 = \
                    self.build_tree(state_m, u, v, j - 1)
            else:
                _, state_p, state_pr_2, n_pr_2, s_pr_2 = \
                    self.build_tree(state_p, u, v, j - 1)

            # Accept.
            sm = n_pr + n_pr_2
            if sm > 0 and self.random.rand() < n_pr_2 / sm:
                state_pr = state_pr_2
            n_pr += n_pr_2

            s_pr = s_pr & s_pr_2 & self.stop_criterion(state_m, state_p)

        return state_m, state_p, state_pr, n_pr, s_pr

    def stop_criterion(self, state_m, state_p):
        delta = state_p.coords - state_m.coords
        return ((np.dot(delta, state_m._momentum) >= 0.0) &
                (np.dot(delta, state_p._momentum) >= 0.0))

    def __call__(self, args):
        state, current_p = args

        # Compute the initial gradient.
        state = self.model.compute_log_probability(state)
        state = self.model.compute_grad_log_probability(state)
        state._momentum = current_p

        # Initialize.
        state_plus = copy.deepcopy(state)
        state_minus = copy.deepcopy(state)
        n = 1

        # Slice sample u.
        f = state.log_probability
        f -= 0.5 * np.dot(current_p, self.cov.apply(current_p))
        u = self.random.uniform(0.0, np.exp(f))
        for j in range(self.max_depth):
            v = 2.0 * (self.random.rand() < 0.5) - 1.0
            if v < 0.0:
                state_minus, _, state_pr, n_pr, s = \
                    self.build_tree(state_minus, u, v, j)
            else:
                _, state_plus, state_pr, n_pr, s = \
                    self.build_tree(state_plus, u, v, j)

            # Accept or reject.
            if s and self.random.rand() < float(n_pr) / n:
                state = state_pr
            n += n_pr

            # Break out after a U-Turn.
            if s and self.stop_criterion(state_minus, state_plus):
                break

        state._nuts_steps = j + 1
        state.accepted = True
        return state, np.inf


class NoUTurnsMove(HamiltonianMove):
    """A HMC move that automatically tunes the number of integration steps.

    This implementations follows `Hoffman & Gelman
    <http://arxiv.org/abs/1111.4246>`_ to tune the number of integration steps
    by watching for "U-turns".

    Args:
        epsilon (float or (2,)): The step size used in the integration. A
            float can be given for a constant step size or a range can be
            given and the final value will be uniformly sampled.
        cov (Optional): An estimate of the parameter covariances. The inverse
            of ``cov`` is used as a mass matrix in the integration. (default:
            ``1.0``)

    """

    _wrapper = _nuts_wrapper

    def __init__(self, epsilon, nsplits=2, cov=1.0):
        self.epsilon = epsilon
        self.nsplits = nsplits
        self.cov = cov

    def get_args(self, ensemble):
        try:
            eps = float(self.epsilon)
        except TypeError:
            eps = ensemble.random.uniform(self.epsilon[0], self.epsilon[1])
        return (eps, )

# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NoUTurnsMove"]

import numpy as np

from ..compat import izip, xrange
from .hmc import HamiltonianMove, _hmc_wrapper


class _nuts_wrapper(_hmc_wrapper):

    def __init__(self, model, cov, epsilon, max_depth=500, delta_max=1000.0):
        self.max_depth = max_depth
        self.delta_max = delta_max
        super(_nuts_wrapper, self).__init__(model, cov, epsilon)

    def leapfrog(self, state, epsilon):
        p = state._momentum + 0.5 * epsilon * state.grad_lnprob
        q = state.coords + epsilon * self.cov.apply(p)
        state = self.model.get_state(q, compute_grad=True)
        state._momentum = p + 0.5 * epsilon * state.grad_lnprob
        return state

    def build_tree(self, state, u, v, j):
        if j == 0:
            state_pr = self.leapfrog(state, v * self.epsilon)
            K_pr = np.dot(state_pr._momentum,
                          self.cov.apply(state_pr._momentum))
            lnprob_pr = state.lnprob - 0.5 * K_pr
            n_pr = int(np.log(u) < lnprob_pr)
            s_pr = np.log(u) - self.delta_max < lnprob_pr
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
            if n_pr_2 > 0.0 and np.random.rand() < n_pr_2 / (n_pr + n_pr_2):
                state_pr = state_pr_2
            n_pr += n_pr_2

            delta = state_p.coords - state_m.coords
            s_pr = s_pr_2 * ((np.dot(delta, state_m._momentum) >= 0.0) &
                             (np.dot(delta, state_p._momentum) >= 0.0))

        return state_m, state_p, state_pr, n_pr, s_pr

    def __call__(self, args):
        state, current_p = args

        # Compute the initial gradient.
        try:
            state.grad_lnprob
        except AttributeError:
            state = self.model.get_state(state.coords, compute_grad=True)
        state._momentum = current_p

        # Initialize.
        state_plus = state
        state_minus = state
        n = 1

        # Slice sample u.
        f = state.lnprob
        f -= 0.5 * np.dot(current_p, self.cov.apply(current_p))
        u = np.random.uniform(0.0, np.exp(f))
        for j in xrange(self.max_depth):
            v = 1.0 - 2.0 * (np.random.rand() < 0.5)
            if v < 0.0:
                state_minus, _, state_pr, n_pr, s = \
                    self.build_tree(state_minus, u, v, j)
            else:
                _, state_plus, state_pr, n_pr, s = \
                    self.build_tree(state_minus, u, v, j)

            # Accept or reject.
            if s and np.random.rand() < float(n_pr) / n:
                state_pr.accepted = True
                state = state_pr
            n += n_pr

            # Break out after a U-Turn.
            delta = state_plus.coords - state_minus.coords
            if (s == 0 or np.dot(delta, state_minus._momentum) < 0.0 or
                    np.dot(delta, state_plus._momentum) < 0.0):
                break

        state.nuts_steps = j
        return state, -np.inf


class NoUTurnsMove(HamiltonianMove):

    _wrapper = _nuts_wrapper

    def __init__(self, epsilon, nsplits=2, cov=1.0):
        self.epsilon = epsilon
        self.nsplits = nsplits
        self.cov = cov

    def get_args(self, ensemble):
        # Randomize the stepsize if requested.
        rand = ensemble.random
        try:
            eps = float(self.epsilon)
        except TypeError:
            eps = rand.uniform(self.epsilon[0], self.epsilon[1])

        return (eps, )

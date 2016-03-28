# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .state import State
from .utils import numerical_gradient_1, numerical_gradient_2

__all__ = ["Model"]


class Model(object):

    def compute_log_prior(self, state, **kwargs):
        """
        Compute the log-prior probability of the model.

        :params coords: the coordinates where the model should be evaluated.

        """
        raise NotImplementedError("this must be implemented by subclasses")

    def compute_grad_log_prior(self, state, **kwargs):
        """
        Compute the gradient of the log-prior probability function of the
        model. By default, this method will compute the gradient numerically
        using a first order method.

        :params coords: the coordinates where the model should be evaluated.

        """
        raise NotImplementedError("this must be implemented by subclasses")

    def compute_log_likelihood(self, state, **kwargs):
        """
        Compute the log-likelihood function of the model.

        :params coords: the coordinates where the model should be evaluated.

        """
        raise NotImplementedError("this must be implemented by subclasses")

    def compute_grad_log_likelihood(self, state, **kwargs):
        """
        Compute the gradient of the log-likelihood function of the model. By
        default, this method will compute the gradient numerically using a
        first order method.

        :params coords: the coordinates where the model should be evaluated.

        """
        raise NotImplementedError("this must be implemented by subclasses")

    def compute_log_probability(self, state, **kwargs):
        """
        Compute the log-probability function of the model. In general, this
        method shouldn't need to be overloaded by users; overload
        :func:`compute_log_prior` and :func:`compute_log_likelihood` instead.

        :params coords: the coordinates where the model should be evaluated.

        """
        # Compute the prior.
        state = self.compute_log_prior(state, **kwargs)
        if not np.isfinite(state.log_prior):
            state.log_prior = -np.inf
            return state

        # Compute the likelihood.
        state = self.compute_log_likelihood(state, **kwargs)
        if not np.isfinite(state.log_likelihood):
            state.log_likelihood = -np.inf
        return state

    def __call__(self, coords, **kwargs):
        state = State(coords)
        return self.compute_log_probability(state, **kwargs)

    def compute_grad_log_probability(self, state, **kwargs):
        """
        Compute the gradient of the log-probability function of the model.
        In general, this method shouldn't need to be overloaded by users;
        overload :func:`compute_grad_log_prior` and
        :func:`compute_grad_log_likelihood` instead.

        :params coords: the coordinates where the model should be evaluated.

        """
        state.grad_log_likelihood = np.zeros(len(state.coords))

        state = self.compute_grad_log_prior(state, **kwargs)
        if not np.all(np.isfinite(state.grad_log_prior)):
            state.grad_log_prior = np.zeros(len(state.coords))
            return state

        state = self.compute_grad_log_likelihood(state, **kwargs)
        if not np.all(np.isfinite(state.grad_log_likelihood)):
            state.grad_log_likelihood = np.zeros(len(state.coords))
        return state


class SimpleModel(Model):
    """

    """

    def __init__(self,
                 log_likelihood_fn,
                 log_prior_fn=None,
                 grad_log_likelihood_fn=None,
                 grad_log_prior_fn=None,
                 args=tuple()):
        if log_prior_fn is None:
            log_prior_fn = _default_log_prior_function
        self.log_prior_fn = log_prior_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.args = args

        if grad_log_prior_fn is None:
            grad_log_prior_fn = numerical_gradient_1(
                self.log_prior_fn, *(self.args)
            )
        self.grad_log_prior_fn = grad_log_prior_fn

        if grad_log_likelihood_fn is None:
            grad_log_likelihood_fn = numerical_gradient_1(
                self.log_likelihood_fn, *(self.args)
            )
        self.grad_log_likelihood_fn = grad_log_likelihood_fn

    def compute_log_prior(self, state, **kwargs):
        state.log_prior = self.log_prior_fn(state.coords,
                                                *(self.args))
        return state

    def compute_grad_log_prior(self, state, **kwargs):
        state.grad_log_prior = self.grad_log_prior_fn(
            state.coords, *(self.args)
        )
        return state

    def compute_log_likelihood(self, state, **kwargs):
        state.log_likelihood = self.log_likelihood_fn(state.coords,
                                                          *(self.args))
        return state

    def compute_grad_log_likelihood(self, state, **kwargs):
        state.grad_log_likelihood = self.grad_log_likelihood_fn(
            state.coords, *(self.args)
        )
        return state

    def check_grad(self, coords, **kwargs):
        """
        Numerically check the gradient computation in :func:`get_grad_lnprob`.
        If "correct" within some tolerance, return ``True``.

        :params coords: the coordinates where the model should be evaluated.

        """
        com_g = (
            self.grad_log_likelihood_fn(coords, *(self.args)) +
            self.grad_log_prior_fn(coords, *(self.args))
        )
        num_g = numerical_gradient_2(self.get_lnprob, coords, **kwargs)
        return np.allclose(com_g, num_g)


def _default_log_prior_function(x, *args):
    return 0.0

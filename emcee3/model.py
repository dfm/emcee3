# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .state import State
from .utils import numerical_gradient_1, numerical_gradient_2

__all__ = ["Model", "SimpleModel"]


class Model(object):
    """An interface to a probabilistic model.

    Subclasses should overload the :func:`Model.compute_log_prior` and
    :func:`Model.compute_log_likelihood` methods to expose the model. To use
    any Hamiltonian moves (:class:`moves.HMCMove`, :class:`moves.NUTSMove`,
    etc.), :func:`Model.compute_grad_log_prior` and
    :func:`Model.compute_grad_log_likelihood` should also be implemented. Each
    of these methods should take a :class:`State` as input, update the relevant
    attributes in place, and return the updated :class:`State`.

    """

    def compute_log_prior(self, state, **kwargs):
        """Compute the log prior probability of the model.

        Since this method is called first, the input ``state`` can only be
        expected to have a ``coords`` attribute. Subclasses should implement
        this method and overwrite the ``log_prior`` attribute on the input
        ``state``.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.

        """
        raise NotImplementedError("'compute_log_prior' must be implemented by "
                                  "subclasses")

    def compute_grad_log_prior(self, state, **kwargs):
        """Compute the gradient of the log prior with respect to coords.

        Subclasses should implement this method and overwrite the
        ``grad_log_prior`` attribute on the input ``state``.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.

        """
        raise NotImplementedError("'compute_grad_log_prior' must be "
                                  "implemented by subclasses")

    def compute_log_likelihood(self, state, **kwargs):
        """Compute the log likelihood of the model.

        This method should always be called after :func:`compute_log_prior`.
        Subclasses should implement this method and overwrite the
        ``log_likelihood`` attribute on the input ``state``.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.

        """
        raise NotImplementedError("'compute_log_likelihood' must be "
                                  "implemented by subclasses")

    def compute_grad_log_likelihood(self, state, **kwargs):
        """Compute the gradient of the log likelihood with respect to coords.

        Subclasses should implement this method and overwrite the
        ``grad_log_likelihood`` attribute on the input ``state``.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.

        """
        raise NotImplementedError("'compute_grad_log_likelihood' must be "
                                  "implemented by subclasses")

    def compute_log_probability(self, state, **kwargs):
        """Compute the log probability of the model.

        Subclasses won't generally need to overload this method. Instead,
        :func:`compute_log_prior` and :func:`compute_log_likelihood` should be
        implemented.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.

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
        """Compute the gradient of the log probability of the model.

        Subclasses won't generally need to overload this method. Instead,
        :func:`compute_grad_log_prior` and :func:`compute_grad_log_likelihood`
        should be implemented.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.

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
    """The simplest modeling interface.

    This model interface wraps functions describing the components of the
    model. At a minimum, a function evaluating the log likelihood (up to a
    constant) must be provided. In this case, the prior function is assumed to
    be uniform and improper. All functions must have the call structure::

        log_likelihood(coords, *args)

    where ``coords`` is a coordinate vector and ``*args`` can be provided
    using the ``args`` keyword argument. The ``log_likelihood`` and
    ``log_prior`` functions should return scalars and the ``grad_*`` functions
    should return ``numpy.array`` objects of the same length as the input
    ``coords``.

    Args:
        log_likelihood (callable): A function that evaluates the log
            likelihood of the model.
        log_prior (Optional[callable]): A function that evaluates the log
            prior of the model. If not provided, this will be assumed to be
            uniform and improper.
        grad_log_likelihood (Optional[callable]): A function that evaluates the
            gradient of the log likelihood of the model. If needed but not
            provided, this will be evaluated numerically using a first order
            method.
        grad_log_prior (Optional[callable]): A function that evaluates the
            gradient of the log prior of the model. If needed but not
            provided, this will be evaluated numerically using a first order
            method.
        args (Optional[tuple]): Any other arguments to be provided to the
            probability functions.

    """

    def __init__(self,
                 log_likelihood,
                 log_prior=None,
                 grad_log_likelihood=None,
                 grad_log_prior=None,
                 args=tuple()):
        # If no prior function is provided, we'll assume it to be flat and
        # improper.
        if log_prior is None:
            log_prior = default_log_prior_function
            grad_log_prior = default_grad_log_prior_function

        self.log_prior_func = log_prior
        self.log_likelihood_func = log_likelihood
        self.args = args

        # By default, numerically compute gradients.
        if grad_log_prior is None:
            grad_log_prior = numerical_gradient_1(
                self.log_prior_func, *(self.args)
            )
        self.grad_log_prior_func = grad_log_prior
        if grad_log_likelihood is None:
            grad_log_likelihood = numerical_gradient_1(
                self.log_likelihood_func, *(self.args)
            )
        self.grad_log_likelihood_func = grad_log_likelihood

    def compute_log_prior(self, state, **kwargs):
        state.log_prior = self.log_prior_func(
            state.coords, *(self.args)
        )
        return state
    compute_log_prior.__doc__ = Model.compute_log_prior.__doc__

    def compute_grad_log_prior(self, state, **kwargs):
        state.grad_log_prior = self.grad_log_prior_func(
            state.coords, *(self.args)
        )
        return state
    compute_grad_log_prior.__doc__ = Model.compute_grad_log_prior.__doc__

    def compute_log_likelihood(self, state, **kwargs):
        state.log_likelihood = self.log_likelihood_func(state.coords,
                                                        *(self.args))
        return state
    compute_log_likelihood.__doc__ = Model.compute_log_likelihood.__doc__

    def compute_grad_log_likelihood(self, state, **kwargs):
        state.grad_log_likelihood = self.grad_log_likelihood_func(
            state.coords, *(self.args)
        )
        return state
    compute_grad_log_likelihood.__doc__ = \
        Model.compute_grad_log_likelihood.__doc__

    def check_grad(self, coords, **kwargs):
        """Check the gradients numerically.

        Args:
            coords (array): The coordinates.

        Returns:
            bool: If the numerical gradients and analytic gradients satisfy
                ``numpy.allclose``.

        """
        com_g = (
            self.grad_log_likelihood_func(coords, *(self.args)) +
            self.grad_log_prior_func(coords, *(self.args))
        )
        num_g = numerical_gradient_2(self.get_lnprob, coords, **kwargs)
        return np.allclose(com_g, num_g)


def default_log_prior_function(x, *args):
    """A uniform improper prior."""
    return 0.0


def default_grad_log_prior_function(x, *args):
    """The gradient of a uniform improper prior."""
    return np.zeros(len(x))

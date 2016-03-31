# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from ..state import State

__all__ = ["HamiltonianMove"]


class _hmc_wrapper(object):

    def __init__(self, random, model, cov, epsilon, nsteps=None):
        self.random = random
        self.model = model
        self.nsteps = nsteps
        self.epsilon = epsilon
        if len(np.atleast_1d(cov).shape) == 2:
            self.cov = _hmc_matrix(np.atleast_2d(cov))
        else:
            self.cov = _hmc_vector(np.asarray(cov))

    def __call__(self, args):
        current_state, current_p = args

        # Sample the initial momentum.
        current_q = current_state.coords
        q = current_q
        p = current_p

        # First take a half step in momentum.
        current_state = self.model.compute_grad_log_probability(current_state)
        p = p + 0.5 * self.epsilon * current_state.grad_log_probability

        # Alternate full steps in position and momentum.
        for i in range(self.nsteps):
            # First, a full step in position.
            q = q + self.epsilon * self.cov.apply(p)

            # Then a full step in momentum.
            if i < self.nsteps - 1:
                state = self.model.compute_grad_log_probability(State(q))
                p = p + self.epsilon * state.grad_log_probability

        # Finish with a half momentum step to synchronize with the position.
        state = self.model.compute_grad_log_probability(State(q))
        p = p + 0.5 * self.epsilon * state.grad_log_probability

        # Negate the momentum. This step really isn't necessary but it doesn't
        # hurt to keep it here for completeness.
        p = -p

        # Compute the log probability of the final state.
        state = self.model.compute_log_probability(state)

        # Automatically reject zero probability states.
        state.accepted = False
        if not np.isfinite(state.log_probability):
            return state, 0.0

        # Compute the acceptance probability factor.
        factor = 0.5 * np.dot(current_p, self.cov.apply(current_p))
        factor -= 0.5 * np.dot(p, self.cov.apply(p))
        return state, factor


class HamiltonianMove(object):
    """A Hamiltonian Monte Carlo move.

    This implementation is based on the algorithm in Figure 2 of Neal (2012;
    http://arxiv.org/abs/1206.1901). By default, gradients of your model are
    computed numerically but this is unlikely to be efficient so it's best if
    you compute the gradients yourself using the
    :func:`Model.compute_grad_log_prior` and
    :func:`Model.compute_grad_log_likelihood` methods.

    Args:
        nsteps (int or (2,)): The number of leapfrog steps to take when
            integrating the dynamics. If an integer is provided, the number of
            steps will be constant. Instead, you can also provide a tuple with
            two integers and these will be treated as lower and upper limits
            on the number of steps and the used value will be uniformly
            sampled within that range.
        epsilon (float or (2,)): The step size used in the integration. Like
            ``nsteps`` a float can be given for a constant step size or a
            range can be given and the final value will be uniformly sampled.
        cov (Optional): An estimate of the parameter covariances. The inverse
            of ``cov`` is used as a mass matrix in the integration. (default:
            ``1.0``)

    """

    _wrapper = _hmc_wrapper

    def __init__(self, nsteps, epsilon, nsplits=2, cov=1.0):
        self.nsteps = nsteps
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

        # Randomize the number of steps.
        try:
            L = int(self.nsteps)
        except TypeError:
            L = rand.randint(self.nsteps[0], self.nsteps[1])

        return eps, L

    def update(self, ensemble):
        # Set up the integrator and sample the initial momenta.
        integrator = self._wrapper(ensemble.random, ensemble.model,
                                   self.cov, *(self.get_args(ensemble)))
        momenta = integrator.cov.sample(ensemble.random, ensemble.nwalkers,
                                        ensemble.ndim)

        # Integrate the dynamics in parallel.
        res = ensemble.pool.map(integrator, zip(ensemble.walkers, momenta))

        # Loop over the walkers and update them accordingly.
        states = []
        for i, (state, factor) in enumerate(res):
            lnpdiff = (
                factor +
                state.log_probability -
                ensemble.walkers[i].log_probability
            )
            if lnpdiff > np.log(ensemble.random.rand()):
                state.accepted = True
            states.append(state)

        ensemble.update(states)
        return ensemble


class _hmc_vector(object):

    def __init__(self, cov):
        self.cov = cov
        self.inv_cov = 1.0 / cov

    def sample(self, random, *shape):
        return random.randn(*shape) * np.sqrt(self.inv_cov)

    def apply(self, x):
        return self.cov * x


class _hmc_matrix(object):

    def __init__(self, cov):
        self.cov = cov
        self.inv_cov = np.linalg.inv(self.cov)

    def sample(self, random, *shape):
        return random.multivariate_normal(np.zeros(shape[-1]),
                                          self.inv_cov,
                                          *(shape[:-1]))

    def apply(self, x):
        return np.dot(self.cov, x)

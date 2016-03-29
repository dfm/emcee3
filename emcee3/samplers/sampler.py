# -*- coding: utf-8 -*-

from __future__ import division, print_function

import logging
import numpy as np
from collections import Iterable

from .. import moves
from ..backends import Backend

__all__ = ["Sampler"]


class Sampler(object):
    """A simple MCMC sampler with a customizable schedule of proposals.

    Args:
        proposals (Optional):

    """

    def __init__(self, proposals=None, backend=None):
        # Save the schedule. This should be a list of proposals.
        if not proposals:
            self._proposals = [moves.StretchMove()]
            self._weights = [1.0]
        elif isinstance(proposals, Iterable):
            try:
                self._proposals, self._weights = zip(*proposals)
            except TypeError:
                self._proposals = proposals
                self._weights = np.ones(len(proposals))
        else:
            self._proposals = [proposals]
            self._weights = [1.0]
        self._weights = np.atleast_1d(self._weights).astype(float)
        self._weights /= np.sum(self._weights)

        # Set up the backend.
        if backend is None:
            self.backend = Backend()
        else:
            self.backend = backend

        # Set the chain to the original untouched state.
        self.reset()

    def reset(self):
        """
        Clear the chain and reset it to its default state.

        """
        self.backend.reset()

    def run(self, ensemble, niter, **kwargs):
        """
        Starting from a given ensemble, run a specific number of steps of
        MCMC. In practice, this method just calls :func:`Sampler.sample` and
        and returns the final ensemble from the iterator.

        :param ensemble:
            The starting :class:`Ensemble`.

        :param niter:
            The number of steps to run.

        :param store: (optional)
            If ``True``, save the chain using the backend. If ``False``,
            reset the backend and don't store anything.

        """
        for ensemble in self.sample(ensemble, niter=niter, **kwargs):
            pass
        return ensemble

    def sample(self, ensemble, niter=None, store=None, thin=1):
        """
        Starting from a given ensemble, start sampling as an iterator yielding
        each updated ensemble.

        :param ensemble:
            The starting :class:`Ensemble`.

        :param niter: (optional)
            The number of steps to run. If not given, the iterator will run
            forever.

        :param store: (optional)
            If ``True``, save the chain using the backend. If ``False``,
            reset the backend and don't store anything.

        :param thin: (optional)
            Only store every ``thin`` step. Note: the backend won't ever know
            about this thinning. Instead, it will just think that the chain
            had only ``niter // thin`` steps. (default: ``1``)

        """
        # Set the default backend behavior if not overridden.
        if niter is not None:
            store = True if store is None else store
        else:
            store = False if store is None else store

        # Warn the user about trying to store the chain without setting the
        # number of iterations.
        if niter is None and store:
            logging.warn("Storing the chain without specifying the total "
                         "number of iterations is very inefficient")

        # Check that the thin keyword is reasonable.
        thin = int(thin)
        assert thin > 0, "Invalid thinning argument"

        # Check the ensemble dimensions.
        if store:
            self.backend.check_dimensions(ensemble)
        else:
            self.backend.reset()

        # Extend the chain to the right length.
        if store:
            if niter is None:
                self.backend.extend(0)
            else:
                self.backend.extend(niter // thin)

        # Start the generator.
        i = 0
        while True:
            # Choose a random proposal.
            p = ensemble.random.choice(self._proposals, p=self._weights)

            # Run the update on the current ensemble.
            ensemble = p.update(ensemble)

            # Store this update if required and if not thinned.
            if (i + 1) % thin == 0:
                if store:
                    self.backend.update(ensemble)
                yield ensemble

            # Finish the chain if the total number of steps was reached.
            i += 1
            if niter is not None and i >= niter:
                return

    def __getattr__(self, attr):
        try:
            return getattr(self.backend, attr)
        except AttributeError:
            raise AttributeError("'Sampler' object has no attribute '{0}'"
                                 .format(attr))

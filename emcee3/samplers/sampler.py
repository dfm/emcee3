# -*- coding: utf-8 -*-

from __future__ import division, print_function

import logging
import numpy as np
from collections import Iterable

from ..moves import StretchMove
from ..backends import Backend

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = ["Sampler"]


class Sampler(object):
    """A simple MCMC sampler with a customizable schedule of moves.

    Args:
        moves (Optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form
            ``[(emcee3.moves.StretchMove(), 0.1), ...]``. When running, the
            sampler will randomly select a move from this list (optionally
            with weights) for each proposal. (default: :class:`StretchMove`)
        backend (Optional): The interface used for saving the samples. By
            default, the samples will be saved to memory using the
            :class:`Backend` interface.

    """

    def __init__(self, moves=None, backend=None):
        # Save the schedule. This should be a list of proposals.
        if not moves:
            self._moves = [StretchMove()]
            self._weights = [1.0]
        elif isinstance(moves, Iterable):
            try:
                self._moves, self._weights = zip(*moves)
            except TypeError:
                self._moves = moves
                self._weights = np.ones(len(moves))
        else:
            self._moves = [moves]
            self._weights = [1.0]
        self._weights = np.atleast_1d(self._weights).astype(float)
        self._weights /= np.sum(self._weights)

        # Set up the backend.
        if backend is None:
            self.backend = Backend()
        else:
            self.backend = backend

    def reset(self):
        """Clear the chain and reset it to its default state."""
        self.backend.reset()

    def run(self, ensemble, niter, progress=False, **kwargs):
        """Run the specified number of iterations of MCMC.

        Starting from a given ensemble, run ``niter`` steps of MCMC. In
        practice, this method just calls :func:`Sampler.sample` and
        and returns the final ensemble from the iterator.

        Args:
            ensemble (Ensemble): The starting :class:`Ensemble`.
            niter (int): The number of steps to run.
            progress (Optional[bool]): Optionally show the sampling progress
                using `tqdm <https://github.com/tqdm/tqdm>`_.
                (default: ``False``)
            **kwargs: Any other arguments are passed to :func:`Sampler.sample`.

        Returns:
            Ensemble: The final state of the ensemble.

        """
        g = self.sample(ensemble, niter=niter, **kwargs)
        if progress:
            if tqdm is None:
                raise ImportError("'tqdm' must be installed to show progress")
            g = tqdm(g, total=niter)
        for ensemble in g:
            pass
        return ensemble

    def sample(self, ensemble, niter=None, store=None, thin=1):
        """Run MCMC iterations and yield each updated ensemble.

        Args:
            ensemble (Ensemble): The starting :class:`Ensemble`.
            niter (Optional[int]): The number of steps to run. If not provided,
                the sampler will run forever.
            store (Optional[bool]): If ``True``, save the chain using the
                backend. If ``False``, reset the backend but don't store
                anything.
            thin (Optional[int]): Only store every ``thin`` step. Note: the
                backend won't ever know about this thinning. Instead, it will
                just think that the chain had only ``niter // thin`` steps.
                (default: ``1``)

        Yields:
            Ensemble: The state of the ensemble at every ``thin``-th step.

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
        if thin <= 0:
            raise ValueError("Invalid thinning argument")

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
            p = ensemble.random.choice(self._moves, p=self._weights)

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

    def get_coords(self, **kwargs):
        return self.backend.get_coords(**kwargs)
    get_coords.__doc__ = Backend.get_coords.__doc__

    def get_log_prior(self, **kwargs):
        return self.backend.get_log_prior(**kwargs)
    get_log_prior.__doc__ = Backend.get_log_prior.__doc__

    def get_log_likelihood(self, **kwargs):
        return self.backend.get_log_likelihood(**kwargs)
    get_log_likelihood.__doc__ = Backend.get_log_likelihood.__doc__

    def get_log_probability(self, **kwargs):
        return self.backend.get_log_probability(**kwargs)
    get_log_probability.__doc__ = Backend.get_log_probability.__doc__

    def get_integrated_autocorr_time(self, **kwargs):
        return self.backend.get_integrated_autocorr_time(**kwargs)
    get_integrated_autocorr_time.__doc__ = \
        Backend.get_integrated_autocorr_time.__doc__

    @property
    def acceptance(self):
        return self.backend.acceptance

    @property
    def acceptance_fraction(self):
        return self.backend.acceptance_fraction

    @property
    def current_coords(self):
        return self.backend.current_coords

    @property
    def coords(self):
        return self.backend.coords

    @property
    def log_prior(self):
        return self.backend.log_prior

    @property
    def log_likelihood(self):
        return self.backend.log_likelihood

    @property
    def log_probability(self):
        return self.backend.log_probability

    def __getattr__(self, attr):
        try:
            return getattr(self.backend, attr)
        except AttributeError:
            raise AttributeError("'Sampler' object has no attribute '{0}'"
                                 .format(attr))

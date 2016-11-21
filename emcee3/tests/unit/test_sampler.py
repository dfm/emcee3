# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from ... import moves, backends, Sampler, Ensemble
from ..common import NormalWalker, TempHDFBackend

__all__ = ["test_schedule", "test_shapes", "test_errors", "test_thin"]


def test_schedule():
    # The default schedule should be a single stretch move.
    s = Sampler()
    assert len(s._moves) == 1
    assert len(s._weights) == 1

    # A single move.
    s = Sampler(moves.GaussianMove(0.5))
    assert len(s._moves) == 1
    assert len(s._weights) == 1

    # A list of moves.
    s = Sampler([moves.StretchMove(), moves.GaussianMove(0.5)])
    assert len(s._moves) == 2
    assert len(s._weights) == 2

    # A weighted list of moves.
    s = Sampler([(moves.StretchMove(), 0.3), (moves.GaussianMove(0.5), 0.1)])
    assert len(s._moves) == 2
    assert len(s._weights) == 2
    assert np.allclose(s._weights, [0.75, 0.25])


def test_shapes():
    run_shapes(backends.Backend())
    with TempHDFBackend() as backend:
        run_shapes(backend)
    run_shapes(backends.Backend(), moves=[moves.GaussianMove(0.5),
                                          moves.DEMove(0.5)])
    run_shapes(backends.Backend(), moves=[(moves.GaussianMove(0.5), 0.1),
                                          (moves.DEMove(0.5), 0.3)])


def run_shapes(backend, moves=None, nwalkers=32, ndim=3, nsteps=100,
               seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble, moves and sampler.
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker(1.), coords, random=rnd)
    sampler = Sampler(moves=moves, backend=backend)

    # Run the sampler.
    ensembles = list(sampler.sample(ensemble, nsteps))
    assert len(ensembles) == nsteps, "wrong number of steps"

    tau = sampler.get_integrated_autocorr_time(c=1, quiet=True)
    assert tau.shape == (ndim,)

    for obj in [sampler, sampler.backend]:
        # Check the shapes.
        assert obj.coords.shape == (nsteps, nwalkers, ndim), \
            "incorrect coordinate dimensions"

        assert obj.log_prior.shape == (nsteps, nwalkers), \
            "incorrect prior dimensions"
        assert obj.log_likelihood.shape == (nsteps, nwalkers), \
            "incorrect likelihood dimensions"
        assert obj.log_probability.shape == (nsteps, nwalkers), \
            "incorrect probability dimensions"

        assert obj.acceptance_fraction.shape == (nwalkers,), \
            "incorrect acceptance fraction dimensions"

        # Check the shape of the flattened coords.
        assert obj.get_coords(flat=True).shape == \
            (nsteps * nwalkers, ndim), "incorrect coordinate dimensions"
        assert obj.get_log_prior(flat=True).shape == \
            (nsteps * nwalkers,), "incorrect prior dimensions"
        assert obj.get_log_likelihood(flat=True).shape == \
            (nsteps*nwalkers,), "incorrect likelihood dimensions"
        assert obj.get_log_probability(flat=True).shape == \
            (nsteps*nwalkers,), "incorrect probability dimensions"

    # This should work (even though it's dumb).
    sampler.reset()
    for i, e in enumerate(sampler.sample(ensemble, store=True)):
        if i >= nsteps - 1:
            break
    assert sampler.coords.shape == (nsteps, nwalkers, ndim), \
        "incorrect coordinate dimensions"
    assert sampler.log_prior.shape == (nsteps, nwalkers), \
        "incorrect prior dimensions"
    assert sampler.log_likelihood.shape == (nsteps, nwalkers), \
        "incorrect likelihood dimensions"
    assert sampler.log_probability.shape == (nsteps, nwalkers), \
        "incorrect probability dimensions"
    assert sampler.acceptance_fraction.shape == (nwalkers,), \
        "incorrect acceptance fraction dimensions"


def test_errors(nwalkers=32, ndim=3, nsteps=5, seed=1234):
    # Set up the random number generator.
    rnd = np.random.RandomState()
    rnd.seed(seed)

    # Initialize the ensemble, proposal, and sampler.
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker(1.0), coords, random=rnd)

    # Test for not running.
    sampler = Sampler()
    with pytest.raises(AttributeError):
        sampler.coords
    with pytest.raises(AttributeError):
        sampler.log_probability

    # What about not storing the chain.
    list(sampler.sample(ensemble, nsteps, store=False))
    with pytest.raises(AttributeError):
        sampler.coords

    # Now what about if we try to continue using the sampler with an ensemble
    # of a different shape.
    list(sampler.sample(ensemble, nsteps))

    coords2 = rnd.randn(nwalkers, ndim+1)
    ensemble2 = Ensemble(NormalWalker(1.), coords2, random=rnd)
    with pytest.raises(ValueError):
        list(sampler.sample(ensemble2, nsteps))

    # Iterating without an end state shouldn't save the chain.
    for i, e in enumerate(sampler.sample(ensemble)):
        if i >= nsteps:
            break
    with pytest.raises(AttributeError):
        sampler.coords


def run_sampler(nwalkers=32, ndim=3, nsteps=25, seed=1234, thin=1):
    rnd = np.random.RandomState()
    rnd.seed(seed)
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(NormalWalker(1.0), coords, random=rnd)
    sampler = Sampler()
    list(sampler.sample(ensemble, nsteps, thin=thin))
    return sampler


def test_thin():
    thinby = 3
    sampler1 = run_sampler()
    sampler2 = run_sampler(thin=thinby)
    for k in ["coords", "log_prior", "log_likelihood", "log_probability"]:
        a = getattr(sampler1, k)[thinby-1::thinby]
        b = getattr(sampler2, k)
        assert np.allclose(a, b), "inconsistent {0}".format(k)

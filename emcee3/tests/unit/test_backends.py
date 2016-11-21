# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from ... import backends, Sampler, Ensemble
from ..common import NormalWalker, TempHDFBackend, MetadataWalker

__all__ = ["test_metadata", "test_hdf", "test_hdf_reload"]


def run_sampler(backend, model=NormalWalker(1.0), nwalkers=32, ndim=3,
                nsteps=5, seed=1234):
    rnd = np.random.RandomState()
    rnd.seed(seed)
    coords = rnd.randn(nwalkers, ndim)
    ensemble = Ensemble(model, coords, random=rnd)
    sampler = Sampler(backend=backend)
    list(sampler.sample(ensemble, nsteps))
    return sampler


def test_metadata():
    sampler1 = run_sampler(backends.Backend(), MetadataWalker(1.0))

    # Check to make sure that the metadata was stored in the right order.
    assert np.allclose(np.mean(sampler1.coords, axis=-1),
                       sampler1.get_value("mean"))
    assert np.allclose(np.mean(sampler1.get_coords(flat=True), axis=-1),
                       sampler1.get_value("mean", flat=True))
    assert np.allclose(np.median(sampler1.coords, axis=-1),
                       sampler1.get_value("median"))
    assert np.allclose(np.median(sampler1.get_coords(flat=True), axis=-1),
                       sampler1.get_value("median", flat=True))

    with TempHDFBackend() as backend:
        sampler2 = run_sampler(backend, MetadataWalker(1.0))

        assert np.allclose(sampler1.get_value("mean"),
                           sampler2.get_value("mean"))
        assert np.allclose(sampler1.get_value("median"),
                           sampler2.get_value("median"))


def test_hdf():
    # Run a sampler with the default backend.
    sampler1 = run_sampler(backends.Backend())

    with TempHDFBackend() as backend:
        sampler2 = run_sampler(backend)

        # Check all of the components.
        for k in ["coords", "log_prior", "log_likelihood", "log_probability",
                  "acceptance_fraction"]:
            a = getattr(sampler1, k)
            b = getattr(sampler2, k)
            assert np.allclose(a, b), "inconsistent {0}".format(k)


def test_hdf_reload():
    with TempHDFBackend() as backend1:
        run_sampler(backend1)

        # Test the state
        state = backend1.random_state
        np.random.set_state(state)

        # Load the file using a new backend object.
        backend2 = backends.HDFBackend(backend1.filename, backend1.name)

        assert state[0] == backend2.random_state[0]
        assert all(np.allclose(a, b)
                   for a, b in zip(state[1:], backend2.random_state[1:]))

        # Check all of the components.
        for k in ["coords", "log_prior", "log_likelihood", "log_probability",
                  "acceptance", "acceptance_fraction"]:
            a = getattr(backend1, k)
            b = getattr(backend2, k)
            assert np.allclose(a, b), "inconsistent {0}".format(k)

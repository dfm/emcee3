# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_hdf", "test_hdf_reload"]

import numpy as np
from ... import backends, Sampler, Ensemble
from ..common import NormalWalker, TempHDFBackend, MetadataWalker


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
    sampler1 = run_sampler(backends.DefaultBackend(), MetadataWalker(1.0))

    # Check to make sure that the metadata was stored in the right order.
    assert np.allclose(np.mean(sampler1.coords, axis=-1),
                       sampler1.get_metadata("mean"))
    assert np.allclose(np.mean(sampler1.get_coords(flat=True), axis=-1),
                       sampler1.get_metadata("mean", flat=True))
    assert np.allclose(np.median(sampler1.coords, axis=-1),
                       sampler1.get_metadata("median"))
    assert np.allclose(np.median(sampler1.get_coords(flat=True), axis=-1),
                       sampler1.get_metadata("median", flat=True))

    with TempHDFBackend() as backend:
        sampler2 = run_sampler(backend, MetadataWalker(1.0))

        assert np.allclose(sampler1.get_metadata("mean"),
                           sampler2.get_metadata("mean"))
        assert np.allclose(sampler1.get_metadata("median"),
                           sampler2.get_metadata("median"))


def test_hdf():
    # Run a sampler with the default backend.
    sampler1 = run_sampler(backends.DefaultBackend())

    with TempHDFBackend() as backend:
        sampler2 = run_sampler(backend)

        # Check all of the components.
        for k in ["coords", "lnprior", "lnlike", "lnprob",
                  "acceptance_fraction"]:
            a = getattr(sampler1, k)
            b = getattr(sampler2, k)
            assert np.allclose(a, b), "inconsistent {0}".format(k)


def test_hdf_reload():
    with TempHDFBackend() as backend1:
        run_sampler(backend1)

        # Load the file using a new backend object.
        backend2 = backends.HDFBackend(backend1.filename, backend1.name)

        # Check all of the components.
        for k in ["coords", "lnprior", "lnlike", "lnprob",
                  "acceptance_fraction"]:
            a = getattr(backend1, k)
            b = getattr(backend2, k)
            assert np.allclose(a, b), "inconsistent {0}".format(k)

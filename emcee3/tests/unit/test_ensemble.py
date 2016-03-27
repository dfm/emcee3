# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from ... import Ensemble
from ..common import NormalWalker, UniformWalker

__all__ = ["test_invalid_coords_init", "test_invalid_dims_init",
           "test_valid_init"]


def test_invalid_coords_init(nwalkers=32, ndim=5, seed=1234):
    # Check for invalid coordinates.
    np.random.seed(seed)
    coords = \
        np.ones(nwalkers)[:, None] + 0.001 * np.random.randn(nwalkers, ndim)
    with pytest.raises(ValueError):
        Ensemble(UniformWalker(), coords)


def test_invalid_dims_init(nwalkers=32, ndim=5, seed=1234):
    # Check for invalid coordinate dimensions.
    np.random.seed(seed)
    coords = np.ones((nwalkers, ndim, 3))
    coords += 0.001 * np.random.randn(*(coords.shape))
    with pytest.raises(ValueError):
        Ensemble(UniformWalker(), coords)


def test_valid_init(nwalkers=32, ndim=5, seed=1234):
    # Check to make sure that valid coordinates work too.
    np.random.seed(seed)
    ivar = np.random.rand(ndim)
    coords = 0.002 * np.random.rand(nwalkers, ndim) - 0.001
    ens = Ensemble(NormalWalker(ivar), coords)
    print(ens.walkers[0])
    assert 0
    assert np.all(np.isfinite(ens.__log_probability__))

# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from ... import moves
from itertools import product
from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_gaussian", "test_uniform_gaussian",
           "test_normal_gaussian_nd"]


@pytest.mark.parametrize("mode,factor", product(
    ["vector", "random", "sequential"], [None, 2.0, 5.0]
))
def test_normal_gaussian(mode, factor, **kwargs):
    _test_normal(moves.GaussianMove(0.5, mode=mode, factor=factor), **kwargs)


@pytest.mark.parametrize("mode,factor", product(
    ["vector", "random", "sequential"], [None, 2.0, 3.0]
))
def test_normal_gaussian_nd(mode, factor, **kwargs):
    ndim = 3
    kwargs["nsteps"] = 4000

    # Isotropic.
    _test_normal(moves.GaussianMove(0.5, factor=factor, mode=mode), ndim=ndim,
                 **kwargs)

    # Axis-aligned.
    _test_normal(moves.GaussianMove(0.5 * np.ones(ndim), factor=factor,
                                    mode=mode), ndim=ndim, **kwargs)
    try:
        _test_normal(moves.GaussianMove(0.5 * np.ones(ndim-1), factor=factor,
                                        mode=mode), ndim=ndim,
                     **kwargs)
    except ValueError:
        pass
    else:
        assert 0, "should raise a ValueError"

    # Full matrix.
    _test_normal(moves.GaussianMove(np.diag(0.5 * np.ones(ndim)),
                                    factor=factor, mode=mode), ndim=ndim,
                 **kwargs)
    try:
        _test_normal(moves.GaussianMove(np.diag(0.5 * np.ones(ndim-1))),
                     ndim=ndim, **kwargs)
    except ValueError:
        pass
    else:
        assert 0, "should raise a ValueError"


@pytest.mark.parametrize("mode,factor", product(
    ["vector", "random", "sequential"], [None, 2.0, 5.0]
))
def test_uniform_gaussian(mode, factor, **kwargs):
    _test_uniform(moves.GaussianMove(0.5, factor=factor, mode=mode), **kwargs)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from ... import moves
from itertools import product
from .test_proposal import _test_normal, _test_uniform

__all__ = ["test_normal_gaussian", "test_uniform_gaussian",
           "test_normal_gaussian_nd"]


def test_normal_gaussian(**kwargs):
    _test_normal(moves.GaussianMove(0.5), **kwargs)


@pytest.mark.parametrize("mode,width", product(
    ["vector", "random", "sequential"], [None, 2.0, 5.0]
))
def test_normal_gaussian_nd(mode, width, **kwargs):
    ndim = 3

    # Isotropic.
    _test_normal(moves.GaussianMove(0.5), ndim=ndim, **kwargs)

    # Axis-aligned.
    _test_normal(moves.GaussianMove(0.5 * np.ones(ndim)), ndim=ndim, **kwargs)
    try:
        _test_normal(moves.GaussianMove(0.5 * np.ones(ndim-1)), ndim=ndim,
                     **kwargs)
    except ValueError:
        pass
    else:
        assert 0, "should raise a ValueError"

    # Full matrix.
    _test_normal(moves.GaussianMove(np.diag(0.5 * np.ones(ndim))), ndim=ndim,
                 **kwargs)
    try:
        _test_normal(moves.GaussianMove(np.diag(0.5 * np.ones(ndim-1))),
                     ndim=ndim, **kwargs)
    except ValueError:
        pass
    else:
        assert 0, "should raise a ValueError"


def test_uniform_gaussian(**kwargs):
    _test_uniform(moves.GaussianMove(0.5), **kwargs)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from ...numgrad import numerical_gradient_1, numerical_gradient_2


__all__ = ["test_numgrad"]


def f(x):
    return -0.5 * np.sum(x**2)


def dfdx(x):
    return -x


@pytest.mark.parametrize("gf", [numerical_gradient_1, numerical_gradient_2])
def test_numgrad(gf, seed=42):
    np.random.seed(seed)

    numgrad = gf(f)

    x = np.zeros(5)
    assert np.allclose(dfdx(x), numgrad(x), atol=2*numgrad.eps)

    x = np.ones(2)
    assert np.allclose(dfdx(x), numgrad(x), atol=2*numgrad.eps)

    x = np.random.randn(7)
    assert np.allclose(dfdx(x), numgrad(x), atol=2*numgrad.eps)

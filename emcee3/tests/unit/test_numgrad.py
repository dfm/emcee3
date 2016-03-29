# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from itertools import product
from ...numgrad import numerical_gradient_1, numerical_gradient_2


__all__ = ["test_numgrad"]


def f1(x):
    return -0.5 * np.sum(x**2)


def dfdx1(x):
    return -x


def f2(x):
    return np.sum(x)


def dfdx2(x):
    return np.ones(len(x))


@pytest.mark.parametrize("funcs,gf", product(
    [(f1, dfdx1), (f2, dfdx2)],
    [numerical_gradient_1, numerical_gradient_2],
))
def test_numgrad(funcs, gf, seed=42):
    f, dfdx = funcs

    np.random.seed(seed)

    numgrad = gf(f)

    x = np.zeros(5)
    assert np.allclose(dfdx(x), numgrad(x), atol=2*numgrad.eps)

    x = np.ones(2)
    assert np.allclose(dfdx(x), numgrad(x), atol=2*numgrad.eps)

    x = np.random.randn(7)
    assert np.allclose(dfdx(x), numgrad(x), atol=2*numgrad.eps)

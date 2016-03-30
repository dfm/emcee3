# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from ...state import State

__all__ = ["test_dtype", "test_serialization", "test_repr"]


def test_dtype(seed=1234):
    np.random.seed(seed)

    dtype = [
        ("coords", np.float64, (4, )),
        ("log_prior", np.float64),
        ("log_likelihood", np.float64),
        ("accepted", bool)
    ]

    coords = np.random.randn(4)
    state = State(coords)
    assert state.dtype == np.dtype(dtype)

    state = State(coords, face=10.0, blah=6, _hidden=None)
    dtype += [
        ("blah", int),
        ("face", float),
    ]
    assert state.dtype == np.dtype(dtype)

    state = State(coords, face=10.0, blah=6, _hidden=None,
                  matrix=np.ones((3, 1)))
    dtype += [
        ("matrix", float, (3, 1)),
    ]
    assert state.dtype == np.dtype(dtype)

    state = State(coords, face=10.0, blah=6, _hidden=None,
                  matrix=np.ones((3, 1)), vector=np.zeros(3))
    dtype += [
        ("vector", float, (3,)),
    ]
    assert state.dtype == np.dtype(dtype)


def test_serialization(seed=1234):
    np.random.seed(seed)
    coords = np.random.randn(4)
    state = State(coords, 0.0, -1.5, True, face="blah")
    array = state.to_array()
    assert np.allclose(array["coords"], coords)
    new_state = State.from_array(array)
    assert state == new_state


def test_repr():
    coords = np.zeros(1)
    lnp = 0.0
    lnl = -1.5
    state = State(coords, lnp, lnl, True)
    assert (
        repr(state) ==
        "State(array({0}), log_prior={1}, log_likelihood={2}, accepted=True)"
        .format(coords, lnp, lnl)
    )

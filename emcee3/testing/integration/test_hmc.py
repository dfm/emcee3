# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_hmc", "test_normal_hmc_nd", "test_adaptive_hmc"]

from ... import moves
from .test_proposal import _test_normal, _test_uniform


def test_normal_hmc(**kwargs):
    _test_normal(moves.HamiltonianMove((10, 20), (0.05, 0.1)), nsteps=100,
                 check_acceptance=False)


def test_uniform_hmc(**kwargs):
    _test_normal(moves.HamiltonianMove((10, 20), (0.05, 0.1)), nsteps=100,
                 check_acceptance=False)


def test_normal_hmc_nd(**kwargs):
    _test_normal(moves.HamiltonianMove(10, 0.1), ndim=3, nsteps=100,
                 check_acceptance=False)


def test_adaptive_hmc(**kwargs):
    _test_normal(moves.HamiltonianMove((10, 20), (0.05, 0.1), cov="adapt"),
                 nsteps=100, check_acceptance=False)

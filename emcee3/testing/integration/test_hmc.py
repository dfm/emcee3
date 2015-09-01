# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_hmc"]

from ... import moves
from .test_proposal import _test_normal


def test_normal_hmc(**kwargs):
    _test_normal(moves.HMCMove(200, 0.05), nsteps=100, check_acceptance=False)

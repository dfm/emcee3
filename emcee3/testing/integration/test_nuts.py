# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_normal_nuts",]

from ... import moves
from .test_proposal import _test_normal, _test_uniform


def test_normal_nuts(**kwargs):
    _test_normal(moves.NoUTurnsMove((0.05, 0.1)), nsteps=200,
                 check_acceptance=False)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
from ... import moves
from .test_proposal import _test_normal

__all__ = ["test_normal_nuts", ]

pytestmark = pytest.mark.skipif()


def test_normal_nuts(**kwargs):
    _test_normal(moves.NoUTurnsMove((0.05, 0.1)), nsteps=200,
                 check_acceptance=False)

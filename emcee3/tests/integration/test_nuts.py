# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
from ... import moves
from .test_proposal import _test_normal

__all__ = ["test_normal_nuts", ]


@pytest.mark.xfail
def test_normal_nuts(**kwargs):
    _test_normal(moves.NoUTurnsMove((1.0, 2.0)), nwalkers=1, nsteps=2000,
                 check_acceptance=False)

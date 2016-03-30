# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from ... import Ensemble
from ..common import NormalWalker, UniformWalker

__all__ = []


def test_simple():
    model = UniformWalker()

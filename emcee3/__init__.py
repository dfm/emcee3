# -*- coding: utf-8 -*-

__version__ = "3.0.0.dev0"

try:
    __EMCEE3_SETUP__
except NameError:
    __EMCEE3_SETUP__ = False

if not __EMCEE3_SETUP__:
    __all__ = ["moves", "pools", "Sampler", "Ensemble",
               "BaseWalker", "SimpleWalker"]

    from . import moves, pools
    from .ensemble import Ensemble
    from .walker import BaseWalker, SimpleWalker
    from .samplers import Sampler

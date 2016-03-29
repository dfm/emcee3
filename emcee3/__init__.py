# -*- coding: utf-8 -*-

__version__ = "3.0.0.dev0"

try:
    __EMCEE3_SETUP__
except NameError:
    __EMCEE3_SETUP__ = False

if not __EMCEE3_SETUP__:
    __all__ = [
        "moves",
        "pools",
        "autocorr",
        "Model",
        "SimpleModel",
        "Sampler",
        "Ensemble",
        "State",
    ]

    from . import moves, pools, autocorr
    from .model import Model, SimpleModel
    from .ensemble import Ensemble
    from .samplers import Sampler
    from .state import State

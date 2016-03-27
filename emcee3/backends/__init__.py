# -*- coding: utf-8 -*-
"""
These backends abstract the storage of and access to emcee3 MCMC chains.

"""

from .backend import Backend
from .hdf import HDFBackend

__all__ = ["Backend", "HDFBackend"]

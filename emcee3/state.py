# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .compat import izip, iteritems

__all__ = ["State"]


class State(object):

    def __init__(self,
                 coords,
                 log_prior=-np.inf,
                 log_likelihood=-np.inf,
                 accepted=False,
                 **metadata):
        self.__coords__ = coords
        self.__log_prior__ = log_prior
        self.__log_likelihood__ = log_likelihood
        self.__accepted__ = accepted
        for k, v in iteritems(metadata):
            setattr(self, k, v)

    def __repr__(self):
        names = self.dtype.names
        values = [
            self.__log_prior__, self.__log_likelihood__, self.__accepted__
        ] + [getattr(self, k) for k in names[4:]]
        names = ["log_prior", "log_likelihood", "accepted"] + list(names[4:])
        r = ", ".join("{0}={1!r}".format(a, b)
                      for a, b in izip(names, values))
        return "State({0!r}, {1})".format(self.__coords__, r)

    @property
    def dtype(self):
        return np.dtype([
            ("__coords__", np.float64, (len(self.__coords__),)),
            ("__log_prior__", np.float64),
            ("__log_likelihood__", np.float64),
            ("__accepted__", bool),
        ] + [(k, np.atleast_1d(v).dtype)
             for k, v in sorted(iteritems(self.__dict__))
             if not k.startswith("_")])

    def to_array(self, out=None):
        if out is None:
            out = np.empty(1, self.dtype)
        for k in out.dtype.names:
            if k.startswith("_"):
                continue
            out[k] = getattr(self, k)
        out["__coords__"] = self.__coords__
        out["__log_prior__"] = self.__log_prior__
        out["__log_likelihood__"] = self.__log_likelihood__
        out["__accepted__"] = self.__accepted__
        return out

    @classmethod
    def from_array(cls, array):
        self = cls(array["__coords__"][0],
                   log_prior=array["__log_prior__"][0],
                   log_likelihood=array["__log_likelihood__"][0],
                   accepted=array["__accepted__"][0])
        for k in array.dtype.names:
            if k.startswith("_"):
                continue
            setattr(self, k, array[k][0])
        return self

    @property
    def __log_probability__(self):
        return self.__log_prior__ + self.__log_likelihood__

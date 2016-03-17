# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np

__all__ = ["numerical_gradient_1", "numerical_gradient_2"]


class numerical_gradient_1(object):

    def __init__(self, f, eps=1.234e-7):
        self.eps = eps
        self.f = f

    def __call__(self, x, *args, **kwargs):
        y0 = self.f(x, *args, **kwargs)
        g = np.zeros(len(x))
        for i, v in enumerate(x):
            x[i] = v + self.eps
            y = self.f(x, *args, **kwargs)
            g[i] = (y - y0) / self.eps
            x[i] = v
        return g


class numerical_gradient_2(object):

    def __init__(self, f, eps=1.234e-7):
        self.eps = eps
        self.f = f

    def __call__(self, x, *args, **kwargs):
        g = np.zeros(len(x))
        for i, v in enumerate(x):
            x[i] = v + self.eps
            yp = self.f(x, *args, **kwargs)
            x[i] = v - self.eps
            ym = self.f(x, *args, **kwargs)
            g[i] = 0.5 * (yp - ym) / self.eps
            x[i] = v
        return g

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import numpy as np

import emcee
import emcee3


def lnprob(x):
    return -0.5 * np.sum(x**2)


N = 5000
coords = np.random.randn(56, 10)

emcee_sampler = emcee.EnsembleSampler(coords.shape[0], coords.shape[1],
                                      lnprob)

strt = time.time()
emcee_sampler.run_mcmc(coords, N)
print("emcee took {0} seconds".format(time.time() - strt))


emcee3_sampler = emcee3.Sampler()
ens = emcee3.Ensemble(emcee3.Model(lnprob), coords)

strt = time.time()
emcee3_sampler.run(ens, N)
print("emcee3 took {0} seconds".format(time.time() - strt))

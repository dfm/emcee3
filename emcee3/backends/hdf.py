# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HDFBackend"]

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .default import DefaultBackend


class HDFBackend(DefaultBackend):

    def __init__(self, filename, name, **kwargs):
        if h5py is None:
            raise ImportError("h5py")
        self.filename = filename
        self.name = name
        super(HDFBackend, self).__init__(**kwargs)

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    def reset(self):
        """
        Clear the chain and reset it to its default state.

        """
        self.initialized = False
        self.nwalkers = None
        self.spec = None

    def extend(self, n):
        k = self.nwalkers
        if not self.initialized:
            with self.open("w") as f:
                g = f.create_group(self.name)
                g.attrs["niter"] = 0
                g.attrs["size"] = n
                for name, shape, dtype in self.spec:
                    g.create_dataset(name, (n, k) + shape, dtype=dtype,
                                     maxshape=(None, k) + shape)
                g.create_dataset("acceptance",
                                 data=np.zeros(k, dtype=np.uint64))

            self.initialized = True

        else:
            with self.open("a") as f:
                g = f[self.name]

                # Update the size entry.
                niter = g.attrs["niter"]
                size = g.attrs["size"]
                l = niter + n
                g.attrs["size"] = size
                for name, _, _ in self.spec:
                    g[name].resize(l, axis=0)

    def update(self, ensemble):
        # Get the current file shape and dimensions.
        with self.open() as f:
            g = f[self.name]
            niter = g.attrs["niter"]
            size = g.attrs["size"]

        # Resize the chain if necessary.
        if niter >= size:
            self.extend(niter - size + 1)

        # Update the file.
        with self.open("a") as f:
            g = f[self.name]
            for w, walker in enumerate(ensemble.walkers):
                for name, _, _ in self.spec:
                    g[name][niter, w] = getattr(walker, name)
            g["acceptance"][:] += ensemble.acceptance
            g.attrs["niter"] = niter + 1

    def get_value(self, name):
        try:
            with self.open() as f:
                g = f[self.name]
                i = g.attrs["niter"]
                return g[name][:i]
        except IOError:
            raise KeyError(name)

    @property
    def niter(self):
        with self.open() as f:
            return f[self.name].attrs["niter"]

    @property
    def acceptance(self):
        with self.open() as f:
            return f[self.name]["acceptance"][...]

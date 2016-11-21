# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pickle
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .backend import Backend

__all__ = ["HDFBackend"]


class HDFBackend(Backend):

    def __init__(self, filename, name="mcmc", **kwargs):
        if h5py is None:
            raise ImportError("h5py")
        self.filename = filename
        self.name = name
        super(HDFBackend, self).__init__(**kwargs)

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    def reset(self):
        """Clear the chain and reset it to its default state.

        """
        self.initialized = False
        super(HDFBackend, self).reset()

    def extend(self, n):
        k = self.nwalkers
        if not self.initialized:
            with self.open("w") as f:
                g = f.create_group(self.name)
                g.attrs["niter"] = 0
                g.attrs["size"] = n
                g.create_dataset("chain", (n, k), dtype=self.dtype,
                                 maxshape=(None, k))
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
                g["chain"].resize(l, axis=0)

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
            for j, walker in enumerate(ensemble.walkers):
                g["chain"][niter, j] = walker.to_array()
            g["acceptance"][:] += ensemble.acceptance
            state = ensemble.random.get_state()
            for i, v in enumerate(state):
                g.attrs["random_state_{0}".format(i)] = v
            g.attrs["niter"] = niter + 1

    def __getitem__(self, name_and_index_or_slice):
        try:
            name, index_or_slice = name_and_index_or_slice
        except ValueError:
            name = name_and_index_or_slice
            index_or_slice = slice(None)

        try:
            with self.open() as f:
                g = f[self.name]
                i = g.attrs["niter"]
                return g["chain"][name][:i][index_or_slice]
        except IOError:
            raise KeyError(name)

    @property
    def niter(self):
        with self.open() as f:
            return f[self.name].attrs["niter"]

    # This no-op is here for compatibility with the default Backend.
    @niter.setter
    def niter(self, value):
        pass

    @property
    def acceptance(self):
        with self.open() as f:
            return f[self.name]["acceptance"][...]

    @property
    def random_state(self):
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

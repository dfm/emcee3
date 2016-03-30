# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

__all__ = ["State"]


class State(object):
    """The current state of a walker.

    This object captures the state of a walker. It will store the coordinates,
    probabilities, and any other computed metadata. Metadata can be added by
    simply adding an attribute to a :class:`State` object. Any attributes with
    names that don't start with an underscore will be serialized by the
    :func:`to_array` method.

    Note:
        Unless the ``accepted`` attribute is ``True``, this object can't
        be expected to have the correct data type.

    Args:
        coords (array[ndim]): The coordinate vector of the walker's state.
        log_prior (Optional[float]): The log prior evaluated at ``coords``. If
            not provided, it is expected that the
            :func:`Model.compute_log_prior` method of a model will save the
            ``log_prior`` attribute on this object.
        log_likelihood (Optional[float]): The log likelihood evaluated at
            ``coords``. Like ``log_prior``, this should be evaluated by the
            model.
        accepted (Optional[bool]): Was this proposal accepted?
        **kwargs: Any other values to store as metadata.

    """

    def __init__(self,
                 coords,
                 log_prior=-np.inf,
                 log_likelihood=-np.inf,
                 accepted=False,
                 **metadata):
        self.coords = coords
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.accepted = accepted
        for k, v in metadata.items():
            setattr(self, k, v)

    def __repr__(self):
        names = self.dtype.names
        values = [
            self.log_prior, self.log_likelihood, self.accepted
        ] + [getattr(self, k) for k in names[4:]]
        r = ", ".join("{0}={1!r}".format(a, b)
                      for a, b in zip(names[1:], values))
        return "State({0!r}, {1})".format(self.coords, r)

    def __eq__(self, other):
        if not self.dtype == other.dtype:
            return False
        return np.all(self.to_array() == other.to_array())

    @property
    def dtype(self):
        base_columns = ["coords", "log_prior", "log_likelihood", "accepted"]
        columns = []
        for k, v in sorted(self.__dict__.items()):
            if k.startswith("_") or k in base_columns:
                continue
            v = np.atleast_1d(v)
            if v.shape == (1,):
                columns.append((k, v.dtype))
            else:
                columns.append((k, v.dtype, v.shape))

        return np.dtype([
            ("coords", np.float64, (len(self.coords),)),
            ("log_prior", np.float64),
            ("log_likelihood", np.float64),
            ("accepted", bool),
        ] + columns)

    def to_array(self, out=None):
        """Serialize the state to a structured numpy array representation.

        This representation will include all attributes of this instance that
        don't have a name beginning with an underscore. There will also always
        be special fields: ``coords``, ``log_prior``, ``log_likelihood``, and
        ``accepted``.

        Args:
            out (Optional[array]): If provided, the state will be serialized
                in place.

        Returns:
            array: The serialized state.

        """
        if out is None:
            out = np.empty(1, self.dtype)
        for k in out.dtype.names:
            if k.startswith("_"):
                continue
            out[k] = getattr(self, k)
        out["coords"] = self.coords
        out["log_prior"] = self.log_prior
        out["log_likelihood"] = self.log_likelihood
        out["accepted"] = self.accepted
        return out

    @classmethod
    def from_array(cls, array):
        """Reconstruct a saved state from a structured numpy array.

        Args:
            array (array): An array produced by serializing a state using the
                :func:`to_array` method.

        Returns:
            State: The reconstructed state.

        """
        self = cls(array["coords"][0],
                   log_prior=array["log_prior"][0],
                   log_likelihood=array["log_likelihood"][0],
                   accepted=array["accepted"][0])
        for k in array.dtype.names:
            if k.startswith("_"):
                continue
            setattr(self, k, array[k][0])
        return self

    @property
    def log_probability(self):
        """A helper attribute that provides access to the log probability.

        """
        return self.log_prior + self.log_likelihood

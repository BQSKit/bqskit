"""This module implements the StateVector class."""
from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import Union

import numpy as np

from bqskit.utils.typing import is_pure_state
from bqskit.utils.typing import is_valid_radixes


class StateVector:
    """The StateVector class."""

    def __init__(self, vec: StateLike, radixes: Sequence[int] = []) -> None:
        """
        Constructs a StateVector with the supplied vector.

        Args:
            vec (StateLike): The state vector.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this StateVector represents. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `vec`.

        Raises:
            TypeError: If `radixes` is not specified and the constructor
                cannot determine `radixes`.

        """
        # Copy Constructor
        if isinstance(vec, StateVector):
            self.vec = vec.get_numpy()
            self.dim = vec.get_dim()
            self.radixes = vec.get_radixes()
            self.size = vec.get_size()
            return

        np_vec = np.array(vec, dtype=np.complex128)

        if not is_pure_state(np_vec):
            raise TypeError('Expected valid state vector.')

        self.vec = np_vec
        self.dim = self.vec.shape[0]

        if radixes:
            self.radixes = tuple(radixes)

        # Check if unitary dimension is a power of two
        elif self.dim & (self.dim - 1) == 0:
            self.radixes = tuple([2] * int(np.round(np.log2(self.dim))))

        # Check if unitary dimension is a power of three
        elif 3 ** int(np.round(np.log(self.dim) / np.log(3))) == self.dim:
            radixes = [3] * int(np.round(np.log(self.dim) / np.log(3)))
            self.radixes = tuple(radixes)

        else:
            raise TypeError(
                'Unable to determine radixes'
                ' for UnitaryMatrix with dim %d.' % self.dim,
            )

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if np.prod(self.radixes) != self.dim:
            raise ValueError('Qudit radixes mismatch with dimension.')

        self.size = len(self.radixes)


StateLike = Union[StateVector, np.ndarray, Sequence[Any]]

"""This module implements the UnitaryBuilder class."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from bqskit.qis.unitary.unitary import Unitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_valid_location
from bqskit.utils.typing import is_valid_radixes

logger = logging.getLogger(__name__)


class UnitaryBuilder(Unitary):
    """
    The UnitaryBuilder Class.

    A UnitaryBuilder is similar to a StringBuilder in the sense that it is an
    efficient way to string together or accumulate Unitary's. This class uses
    concepts from tensor networks to efficiently multiply unitary matrices.
    """

    def __init__(self, num_qudits: int, radixes: Sequence[int] = []) -> None:
        """
        UnitaryBuilder constructor.

        Args:
            num_qudits (int): The number of qudits to build a Unitary for.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: if num_qudits is nonpositive.

        Examples:
            >>> builder = UnitaryBuilder(4)  # Creates a 4-qubit builder.
        """

        if not isinstance(num_qudits, int):
            raise TypeError(
                'Invalid type for num_qudits: '
                'expected int, got %s.' % type(num_qudits),
            )

        if num_qudits <= 0:
            raise ValueError('Expected positive number for num_qudits.')

        self.size = num_qudits
        self.num_params = 0
        self.radixes = tuple(radixes or [2] * num_qudits)

        if not is_valid_radixes(self.radixes, self.get_size()):
            raise TypeError('Invalid qudit radixes.')

        self.tensor = np.identity(self.get_dim())
        self.tensor = self.tensor.reshape(self.radixes * 2)

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Build the unitary."""
        utry = self.tensor.reshape((self.get_dim(), self.get_dim()))
        return UnitaryMatrix(utry, self.get_radixes())

    def apply_right(
        self, utry: UnitaryMatrix,
        location: Sequence[int], inverse: bool = False,
    ) -> None:
        """
        Apply the specified unitary on the right of this UnitaryBuilder.

             .-----.   .------.
          0 -|     |---|      |-
          1 -|     |---| utry |-
             .     .   '------'
             .     .
             .     .
        n-1 -|     |------------
             '-----'

        Args:
            utry (UnitaryMatrix): The unitary to apply.

            location (Sequence[int]): The qudits to apply the unitary on.

            inverse (bool): If true, apply the inverse of the unitary.

        Notes:
            Applying the unitary on the right is equivalent to multiplying
            the unitary on the left of the tensor. This operation is
            performed using tensor contraction.
        """

        if not isinstance(utry, UnitaryMatrix):
            raise TypeError('Expected UnitaryMatrix, got %s', type(utry))

        if not is_valid_location(location, self.get_size()):
            raise TypeError('Invalid location.')

        if len(location) != utry.get_size():
            raise ValueError('Unitary and location size mismatch.')

        for utry_radix, bldr_radix_idx in zip(utry.get_radixes(), location):
            if utry_radix != self.get_radixes()[bldr_radix_idx]:
                raise ValueError('Unitary and location radix mismatch.')

        left_perm = list(location)
        mid_perm = [x for x in range(self.get_size()) if x not in location]
        right_perm = [x + self.get_size() for x in range(self.get_size())]

        left_dim = int(np.prod([self.get_radixes()[x] for x in left_perm]))

        utry = utry.get_dagger() if inverse else utry
        utry_np = utry.get_numpy()

        perm = left_perm + mid_perm + right_perm
        self.tensor = self.tensor.transpose(perm)
        self.tensor = self.tensor.reshape((left_dim, -1))
        self.tensor = utry_np @ self.tensor

        shape = list(self.get_radixes()) * 2
        shape = [shape[p] for p in perm]
        self.tensor = self.tensor.reshape(shape)
        inv_perm = list(np.argsort(perm))
        self.tensor = self.tensor.transpose(inv_perm)

    def apply_left(
        self, utry: UnitaryMatrix,
        location: Sequence[int], inverse: bool = False,
    ) -> None:
        """
        Apply the specified unitary on the left of this UnitaryBuilder.

             .------.   .-----.
          0 -|      |---|     |-
          1 -| gate |---|     |-
             '------'   .     .
                        .     .
                        .     .
        n-1 ------------|     |-
                        '-----'

        Args:
            utry (UnitaryMatrix): The unitary to apply.

            location (Sequence[int]): The qudits to apply the unitary on.

            inverse (bool): If true, apply the inverse of the unitary.

        Notes:
            Applying the unitary on the left is equivalent to multiplying
            the unitary on the right of the tensor. This operation is
            performed using tensor contraction.
        """

        if not isinstance(utry, UnitaryMatrix):
            raise TypeError('Expected UnitaryMatrix, got %s', type(utry))

        if not is_valid_location(location, self.get_size()):
            raise TypeError('Invalid location.')

        if len(location) != utry.get_size():
            raise ValueError('Unitary and location size mismatch.')

        for utry_radix, bldr_radix_idx in zip(utry.get_radixes(), location):
            if utry_radix != self.get_radixes()[bldr_radix_idx]:
                raise ValueError('Unitary and location radix mismatch.')

        left_perm = list(range(self.get_size()))
        mid_perm = [
            x + self.get_size()
            for x in left_perm if x not in location
        ]
        right_perm = [x + self.get_size() for x in location]

        right_dim = int(
            np.prod([
                self.get_radixes()[x - self.get_size()]
                for x in right_perm
            ]),
        )

        utry = utry.get_dagger() if inverse else utry
        utry_np = utry.get_numpy()

        perm = left_perm + mid_perm + right_perm
        self.tensor = self.tensor.transpose(perm)
        self.tensor = self.tensor.reshape((-1, right_dim))
        self.tensor = self.tensor @ utry_np

        shape = list(self.get_radixes()) * 2
        shape = [shape[p] for p in perm]
        self.tensor = self.tensor.reshape(shape)
        inv_perm = list(np.argsort(perm))
        self.tensor = self.tensor.transpose(inv_perm)

    def calc_env_matrix(self, location: Sequence[int]) -> np.ndarray:
        """
        Calculates the environmental matrix w.r.t. the specified location.

        Args:
            location (Sequence[int]): Calculate the env_matrix with respect
                to the qudit indices in location.

        Returns:
            (np.ndarray): The environmental matrix.
        """

        left_perm = list(range(self.get_size()))
        left_perm = [x for x in left_perm if x not in location]
        left_perm = left_perm + [x + self.get_size() for x in left_perm]
        right_perm = list(location) + [x + self.get_size() for x in location]

        perm = left_perm + right_perm
        a = np.transpose(self.tensor, perm)
        a = np.reshape(
            a, (
                2 ** (self.get_size() - len(location)),
                2 ** (self.get_size() - len(location)),
                2 ** len(location),
                2 ** len(location),
            ),
        )
        return np.trace(a)  # type: ignore
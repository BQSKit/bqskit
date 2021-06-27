"""
This module implements the PermutationMatrix and some helper functions.

A PermutationMatrix is a binary, square matrix with a single 1 in each row and
column.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from bqskitrs import calc_permutation_matrix

from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_permutation
from bqskit.utils.typing import is_valid_location


class PermutationMatrix(UnitaryMatrix):
    """The PermutationMatrix class."""

    def __init__(self, perm: UnitaryLike) -> None:
        """PermutationMatrix Constructor."""
        np_perm = np.array(perm, dtype=np.complex128)

        if not is_permutation(np_perm):
            raise TypeError('Invalid permutation matrix.')

        super().__init__(np_perm)

    @staticmethod
    @lru_cache(maxsize=None)
    def from_qubit_location(
        num_qubits: int,
        location: Sequence[int],
    ) -> PermutationMatrix:
        """
        Creates the permutation matrix specified by arguments.

        The resulting matrix will move the first len(location) qubits
        into positions defined by location.

        Args:
            num_qubits (int): Total number of qubits

            location (Sequence[int]): The desired locations to swap
                the starting qubits to.

        Returns:
            (PermutationMatrix): A 2**num_qubits by 2**num_qubits permutation
                matrix.

        Examples:
            calc_permutation_matrix( 2, (0, 1) ) =
                [ [ 1, 0, 0, 0 ],
                  [ 0, 1, 0, 0 ],
                  [ 0, 0, 1, 0 ],
                  [ 0, 0, 0, 1 ] ]

            Here the 4x4 identity is returned because there are 2 total
            qubits, specified by the first parameter, and the desired
            permutation is [0, 1] -> [0, 1].

            calc_permutation_matrix( 2, (1,) ) = # Also equals
            calc_permutation_matrix( 2, (1, 0) ) =
                [ [ 1, 0, 0, 0 ],
                [ 0, 0, 1, 0 ],
                [ 0, 1, 0, 0 ],
                [ 0, 0, 0, 1 ] ]
            This is a more interesting example. The swap gate is returned
            here since we are working with 2 qubits and want the permutation
            that swaps the two qubits, giving by the permutation [0] -> [1]
            or in the second case [0, 1] -> [1, 0]. Both calls produce
            identical permutations.
        """

        if not isinstance(num_qubits, int):
            raise TypeError(
                'Expected integer num_qudits'
                ', got %s.' % type(num_qubits),
            )

        if num_qubits <= 0:
            raise ValueError(
                'Expected positive num_qudits'
                ', got %d.' % num_qubits,
            )

        if not is_valid_location(location, num_qubits):
            raise TypeError('Invalid location.')

        matrix = calc_permutation_matrix(num_qubits, location)
        return PermutationMatrix(matrix)

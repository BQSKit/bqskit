"""
This module implements the PermutationMatrix and some helper functions.

A PermutationMatrix is a binary, square matrix with a single 1 in each row and
column.

"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import sympy.combinatorics as cb

from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_permutation
from bqskit.utils.typing import is_valid_location


def swap_bit(i: int, j: int, b: int) -> int:
    """
    Swaps bits i and j in b.

    Args:
        i (int): Bit index 1

        j (int): Bit index 2

        b (int): Input number

    Returns:
        (int): The number b with the bits in index 1 and 2 swapped.

    """

    if i == j:
        return b

    b_i = (b >> i) & 1
    b_j = (b >> j) & 1

    if b_i != b_j:
        b &= ~((1 << i) | (1 << j))
        b |= (b_i << j) | (b_j << i)

    return b


def swap(x: int, y: int, n: int) -> cb.Permutation:
    """
    Returns a permutation for the swap between qubits x and y.

    Args:
        x (int): Qubit index 1

        y (int): Qubit index 2

        n (int): Total number of qubits

    Returns:
        (cb.Permutation): The permutation that swaps qubits x and y.

    Raises:
        ValueError: If x or y is an invalid qubit index.

    """

    if x < 0 or x > n or y < 0 or y > n:
        raise ValueError('Invalid qubit index.')

    if x == y:
        return cb.Permutation(2 ** n)

    return cb.Permutation([
        swap_bit(n - 1 - x, n - 1 - y, b)
        for b in range(2 ** n)
    ])


class PermutationMatrix(UnitaryMatrix):
    """The PermutationMatrix class."""

    def __init__(self, perm: UnitaryLike) -> None:
        """PermutationMatrix Constructor."""
        np_perm = np.array(perm, dtype=np.complex128)

        if not is_permutation(np_perm):
            raise TypeError('Invalid permutation matrix.')

        super().__init__(np_perm)

    @staticmethod
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

            location (Tuple[int]): The desired locations to swap
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

        max_qubit = np.max(location)
        num_core_qubits = max_qubit + 1
        num_gate_qubits = len(location)

        perm = cb.Permutation(2**num_core_qubits)
        temp_pos = list(range(num_gate_qubits))

        for q in range(num_gate_qubits):
            perm *= swap(temp_pos[q], location[q], num_core_qubits)

            if location[q] < num_gate_qubits:
                temp_pos[location[q]] = temp_pos[q]

        matrix = np.identity(2 ** num_core_qubits)

        for transpos in reversed(perm.transpositions()):
            matrix[list(transpos), :] = matrix[list(reversed(transpos)), :]

        if num_qubits - num_core_qubits > 0:
            matrix = np.kron(
                matrix, np.identity(
                    2 ** (num_qubits - num_core_qubits),
                ),
            )

        return PermutationMatrix(matrix)

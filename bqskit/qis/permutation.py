"""This module implements the PermutationMatrix class."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
_logger = logging.getLogger(__name__)


class PermutationMatrix(UnitaryMatrix):
    """A binary, unitary matrix with a single 1 in each row and column."""

    def __init__(
        self,
        input: UnitaryLike,
        radixes: Sequence[int] = [],
        check_arguments: bool = True,
    ) -> None:
        """
        Constructs a `PermutationMatrix` from the supplied matrix.

        See :class:`UnitaryMatrix` for more info.
        """
        if check_arguments and not PermutationMatrix.is_permutation(input):
            raise ValueError('Input failed permutation condition.')

        super().__init__(input, radixes, check_arguments)

    swap_utry = UnitaryMatrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

    @staticmethod
    def from_qubit_location(
        num_qubits: int,
        location: Sequence[int],
    ) -> PermutationMatrix:
        """
        Creates the permutation matrix specified by arguments. The resulting
        matrix will move the location's qubits into positions defined by their
        index in location.

        P = Permutationmatrix.from_qubit_location(3, (1, 2, 0))

                .-----.
             0 -|     |- 1
             1 -|  P  |- 2
             2 -|     |- 0
                '-----'

        Args:
            num_qubits (int): Total number of qubits

            location (Sequence[int]): The desired qubits to swap
                to their index.

        Returns:
            (PermutationMatrix): A 2**num_qubits by 2**num_qubits
                permutation matrix.
        """
        current_perm = list(location)
        for i in range(num_qubits):
            if i not in current_perm:
                current_perm.append(i)

        perm_builder = UnitaryBuilder(num_qubits)
        for index, qudit in enumerate(current_perm):
            if index != qudit:
                current_pos = current_perm.index(index)
                perm_builder.apply_left(
                    PermutationMatrix.swap_utry, (index, current_pos),
                )
                tmp = current_perm[index]
                current_perm[index] = current_perm[current_pos]
                current_perm[current_pos] = tmp
        return PermutationMatrix(perm_builder.get_unitary())

    @staticmethod
    def is_permutation(P: np.typing.ArrayLike, tol: float = 1e-8) -> bool:
        """
        Check if P is a permutation matrix.

        Args:
            P (np.typing.ArrayLike): The matrix to check.

            tol (float): The numerical precision of the check.

        Returns:
            bool: True if P is a permutation matrix.
        """

        if isinstance(P, PermutationMatrix):
            return True

        if not UnitaryMatrix.is_unitary(P, tol):
            return False

        if not isinstance(P, np.ndarray):
            P = np.array(P)

        if not all(s == 1 for s in P.sum(0)):
            _logger.debug('Not all rows sum to 1.')
            return False

        if not all(s == 1 for s in P.sum(1)):
            _logger.debug('Not all columns sum to 1.')
            return False

        if not all(e == 1 or e == 0 for row in P for e in row):
            _logger.debug('Not all elements are 0 or 1.')
            return False

        return True

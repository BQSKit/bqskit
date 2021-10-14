"""This module implements the PermutationMatrix class."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
from bqskitrs import calc_permutation_matrix

from bqskit.ir.location import CircuitLocation
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

    @staticmethod
    @lru_cache(maxsize=None)
    def from_qubit_location(
        num_qubits: int,
        location: tuple[int, ...],
    ) -> PermutationMatrix:
        """
        Creates the permutation matrix specified by arguments.

        The resulting matrix will move the first len(location) qubits
        into positions defined by location.

        Args:
            num_qubits (int): Total number of qubits

            location (tuple[int, ...]): The desired locations to swap
                the starting qubits to.

        Returns:
            PermutationMatrix: A 2**num_qubits by 2**num_qubits permutation
            matrix.

        Examples:
            >>> calc_permutation_matrix( 2, (0, 1) )
                [ [ 1, 0, 0, 0 ],
                  [ 0, 1, 0, 0 ],
                  [ 0, 0, 1, 0 ],
                  [ 0, 0, 0, 1 ] ]

            Here the 4x4 identity is returned because there are 2 total
            qubits, specified by the first parameter, and the desired
            permutation is [0, 1] -> [0, 1].

            >>> calc_permutation_matrix( 2, (1,) ) = # Also equals
            >>> calc_permutation_matrix( 2, (1, 0) ) =
                [ [ 1, 0, 0, 0 ],
                  [ 0, 0, 1, 0 ],
                  [ 0, 1, 0, 0 ],
                  [ 0, 0, 0, 1 ] ]

            This is a more interesting example. The swap gate's unitary
            is returned here since we are working with 2 qubits and want
            the permutation that swaps the both qubits. This is given by
            the permutation [0] -> [1] and [0, 1] -> [1, 0] in the second
            case. Both calls produce identical permutations.
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

        if not CircuitLocation.is_location(location, num_qubits):
            raise TypeError('Invalid location.')

        matrix = calc_permutation_matrix(num_qubits, location)
        return PermutationMatrix(matrix)

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

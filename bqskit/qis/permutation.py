"""This module implements the PermutationMatrix class."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
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
    def from_qubit_location(
        num_qubits: int,
        location: Sequence[int],
    ) -> PermutationMatrix:
        """
        Creates a qubit permutation matrix.

        See :func:`from_qudit_location` for more info.
        """
        return PermutationMatrix.from_qudit_location(num_qubits, 2, location)

    @staticmethod
    def from_qudit_location(
        num_qudits: int,
        radix: int,
        location: Sequence[int],
    ) -> PermutationMatrix:
        """
        Creates the permutation matrix specified by arguments.

        The resulting matrix will move the location's qudits into positions
        defined by their index in location. For example:

        P = Permutationmatrix.from_qudit_location(3, 2, (1, 2, 0))

                .-----.
             0 -|     |- 1
             1 -|  P  |- 2
             2 -|     |- 0
                '-----'

        Args:
            num_qudits (int): Total number of qudits.

            radix (int): The base of all qudits.

            location (Sequence[int]): The desired qudits to swap
                to their index.

        Returns:
            (PermutationMatrix): A permutation matrix that moves the
                location's qudits into positions defined by their index
                in location.
        """
        swap_utry = PermutationMatrix.gen_swap_unitary(radix)
        current_perm = list(location)
        for i in range(num_qudits):
            if i not in current_perm:
                current_perm.append(i)

        perm_builder = UnitaryBuilder(num_qudits, [radix] * num_qudits)
        for index, qudit in enumerate(current_perm):
            if index != qudit:
                current_pos = current_perm.index(index)
                perm_builder.apply_left(swap_utry, (index, current_pos))
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

    @staticmethod
    def gen_swap_unitary(radix: int = 2) -> UnitaryMatrix:
        """
        Generate a unitary matrix that swaps the state of two qudits.

        Args:
            radix (int): The base of the qudits being swapped.
                Defaults to qubits or base 2. (Default: 2)

        Raises:
            ValueError: If radix is less than two.
        """
        if not is_integer(radix):
            raise TypeError('Expected a single integer radix.')

        if radix < 2:
            raise ValueError('Radix must be at least 2.')

        dim = radix * radix
        mat = [[0 for _j in range(dim)] for _i in range(dim)]
        for col in range(dim):
            # col = a * radix + b; a, b < radix
            a = col // radix
            b = col % radix
            row = b * radix + a
            mat[row][col] = 1

        return UnitaryMatrix(mat, [radix, radix])

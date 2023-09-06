"""This module implements the SubSwapGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class SubSwapGate(ConstantGate, QuditGate):
    """
    The two-qudit subspace SWAP gate.

    The subspace SWAP gate swaps between "qudit-levels" on a two-qudit gate. For
    example, a |01> to |20> swap would be the identity with the |01> row and
    |20> rows swapped.
    """

    _num_qudits = 2
    _num_params = 0

    def __init__(self, radix: int, qudit_levels: str) -> None:
        """
        Construct a SubSwapGate.

        Args:
            radix (int): The number of qudit levels (>=2).

            qudit_levels (str): The qudit levels that should be swapped,
                separated by a comma. Example: "0,1;2,0" to swap |01>
                to |20>

        Raises:
            ValueError: if radix < 2

            ValueError: If any of the qudit levels integer represenation
                greater than or equal to radix.

            ValueError: If the qudit_levels string is not formatted correctly.
        """

        if not is_integer(radix):
            raise TypeError(f'Expected integer for radix, got {type(radix)}.')

        if radix < 2:
            raise ValueError(f'Radix must be greater than 1, got {radix}.')

        if not isinstance(qudit_levels, str):
            raise TypeError(
                'Expected qudit_levels object to be string'
                f', got {type(qudit_levels)}.',
            )

        self._radix = radix
        level1, level2 = self.decode_qudit_level_string(qudit_levels)
        self._utry = self.calculate_level_swap_unitary(radix, level1, level2)

    @staticmethod
    def calculate_level_swap_unitary(
        radix: int,
        level1: tuple[int, int],
        level2: tuple[int, int],
    ) -> UnitaryMatrix:
        """
        Calculate the unitary for a qudit level swap.

        Args:
            radix (int): The number of qudit levels (>=2).

            level1 (tuple[int, int]): The first qudit level to swap.

            level2 (tuple[int, int]): The second qudit level to swap.

        Returns:
            UnitaryMatrix: The unitary matrix for the qudit level swap.

        Raises:
            ValueError: If any of the qudit levels integer represenation
                greater than or equal to radix.

        Example:
            >>> from bqskit.ir.gates.constant.subswap import SubSwapGate
            >>> u = SubSwapGate.calculate_level_swap_unitary(2, (1, 1), (0, 1))
            >>> u
            UnitaryMatrix([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]], [2, 2])
            >>> from bqskit.ir.gates.constant.cx import CNOTGate
            >>> u == CNOTGate().get_unitary()
        """
        if any(l >= radix for l in level1 + level2):
            raise ValueError(
                'Qudit levels must be less than radix'
                f', got |{level1}> swaps |{level2}> when {radix=}.',
            )

        utry = np.identity(radix ** 2, dtype=np.complex128)
        i = level1[0] * radix + level1[1]
        j = level2[0] * radix + level2[1]
        utry[i, i] = 0
        utry[j, j] = 0
        utry[i, j] = 1
        utry[j, i] = 1
        return UnitaryMatrix(utry, [radix, radix])

    @staticmethod
    def decode_qudit_level_string(
        string: str,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Decode the qudit level string into two pairs of integers.

        Args:
            string (str): The qudit level string to decode. See the
                :class:`SubSwapGate` documentation for more info on the
                format.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: The two two-qudit
                levels to swap, i.e., "0,1;2,0" to swap |01> to |20>.

        Raises:
            ValueError: See :class:`SubSwapGate` documentation for more info.
        """
        split_values = string.split(';', 1)

        if len(split_values) != 2:
            raise ValueError(
                'Qudit levels string must contain exactly one semicolon,'
                f' got {string}.',
            )

        first = split_values[0].split(',')
        second = split_values[1].split(',')

        if len(first) != 2 or len(second) != 2:
            raise ValueError(
                'Qudit levels string must contain exactly one comma in each'
                f' section, got {string}.',
            )

        return (int(first[0]), int(first[1])), (int(second[0]), int(second[1]))

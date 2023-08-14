"""This module implements the SubSwapGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class SubSwapGate(QuditGate): #TODO fix docs
    r"""
    The two-qudit subspace SWAP gate.

    The subspace SWAP gate swaps between "qudit-levels"
    on a two-qudit gate.
    For example, a |01> to |20> swap would be the identity
    with the |01> row and |20> rows swapped.

    __init__() arguments:
        num_levels : int
            Number of levels in each qudit (d).
        qudit_levels : str
            The qudit levels that should be swapped, separated by a comma.
            Example: "0,1;2,0" to swap |01> to |20>
    """

    _num_qudits = 2
    _num_params = 0

    def __init__(
        self, 
        num_levels: int, 
        qudit_levels: str
    ):
        """

        Raises:
            TypeError: If num_levels is not of type int
            TypeError: If qudit_levels is not of type str
            
            ValueError: If any of the qudit levels integer represenation greater than or equal to num_levels
        """
        if not is_integer(num_levels):
            raise TypeError('Expected num_levels object to be integer, got %s.' % type(num_levels))
        
        if type(qudit_levels)!=str:
            raise TypeError('Expected qudit_levels object to be string, got %s.' % type(qudit_levels))

        self.num_levels = num_levels
        level1, level2 = self.convert_string_to_lists(qudit_levels)

        if np.any(np.array(level1)>=num_levels) or np.any(np.array(level2)>=num_levels):
            raise ValueError('Level1 and level2 must not contain any element greater than or equal to num_levels.')

        self.qudit_level1 = level1
        self.qudit_level2 = level2

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        # qubit level indices |ival,jval>
        ival = 0
        jval = 0

        # unitary matrix
        matrix = np.zeros([self.num_levels**2, self.num_levels**2])

        # building the matrix column by column
        for i, col in enumerate(matrix.T):

            # checking to see if the column is one that should be swapped
            # and if so, doing the swap
            if ival == self.qudit_level1[0] and jval == self.qudit_level1[1]:
                iswap = self.qudit_level2[0]
                jswap = self.qudit_level2[1]
                pos = self.num_levels * jswap + iswap
            elif ival == self.qudit_level2[0] and jval == self.qudit_level2[1]:
                iswap = self.qudit_level1[0]
                jswap = self.qudit_level1[1]
                pos = self.num_levels * jswap + iswap
            else:
                pos = self.num_levels * jval + ival
            col[pos] = 1
            matrix[:, i] = col

            # updating ival and jval
            if ival == self.num_levels - 1:
                ival = 0
                jval += 1
            else:
                ival += 1
        u_mat = UnitaryMatrix(matrix, self.radixes)
        return u_mat

    @staticmethod
    def convert_string_to_lists(string: str) -> tuple[list[int], list[int]]:
        split_values = string.split(';')
        list1: list[int] = []
        list2: list[int] = []
        for i, values in enumerate(split_values):
            numbers = values.split(',')
            if i == 0:
                list1.append(int(numbers[1]))
                list1.append(int(numbers[0]))
            else:
                list2.append(int(numbers[1]))
                list2.append(int(numbers[0]))
        return list1, list2

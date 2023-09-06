"""This module implements the CSUMDGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class CSUMGate(ConstantGate, QuditGate):
    """
    The two-qudit Conditional-SUM gate.

    The CSUM gate is given by the following formula:

    .. math::
        \\begin{equation}
            CSUM |i,j> = |i, i + j mod d>
        \\end{equation}

    where d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)


    For qutrits the CSUMGate is represented by the following matrix:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
        \\end{pmatrix}

    References:
        - https://www.frontiersin.org/articles/10.3389/fphy.2020.589504/full
        - https://pubs.aip.org/aip/jmp/article-abstract/56/3/032202/763827
    """

    _num_qudits = 2
    _num_params = 0

    def __init__(self, radix: int = 3) -> None:
        """
        Construct a CSUMGate.

        Args:
            radix (int): The number of qudit levels (>=2). Defaults to
                qutrits.

        Raises:
            ValueError: if radix < 2
        """
        if not is_integer(radix):
            raise TypeError(f'Expected integer for radix, got {type(radix)}.')

        if radix < 2:
            raise ValueError(f'Radix must be greater than 1, got {radix}.')

        self._radix = radix

        ival = 0
        jval = 0
        matrix = np.zeros([self.radix**2, self.radix**2])
        for i, col in enumerate(matrix.T):
            col[self.radix * jval + ((ival + jval) % self.radix)] = 1
            matrix[:, i] = col
            if ival == self.radix - 1:
                ival = 0
                jval += 1
            else:
                ival += 1
        self._utry = UnitaryMatrix(matrix, self.radixes)

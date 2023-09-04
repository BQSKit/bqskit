"""This module implements the ShiftGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class ShiftGate(ConstantGate, QuditGate):
    """
    The one-qudit shift (X) gate. This is a Weyl-Heisenberg gate.

    This gate shifts the state of a qudit up by one level modulo. For
    example, the shift gate on a qubit is the Pauli-X gate. The shift
    gate on a qutrit is the following matrix:

    .. math::

        \\begin{pmatrix}
        0 & 0 & 1 \\\\
        1 & 0 & 0 \\\\
        0 & 1 & 0 \\\\
        \\end{pmatrix}


    The shift gate is generally given by the following formula:

    .. math::
        \\begin{equation}
            X = \\sum_a |a + 1 mod d ><a|
        \\end{equation}

    where d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    References:
        - https://arxiv.org/pdf/2302.07966.pdf
    """

    _num_qudits = 1

    def __init__(self, radix: int = 2) -> None:
        """
        Construct a ShiftGate.

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

        # Calculate unitary
        matrix = np.zeros([radix, radix])
        for j in range(radix):
            matrix[(j + 1) % radix, j] = 1
        self._utry = UnitaryMatrix(matrix, self.radixes)

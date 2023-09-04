"""This module implements the ClockGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class ClockGate(ConstantGate, QuditGate):
    """
    The one-qudit clock (Z) gate. This is a Weyl-Heisenberg gate.

    The clock gate is given by the following formula:

    .. math::
        \\begin{equation}
            Z = \\sum_a \\exp(2\\pi ia/d) |a><a|
        \\end{equation}

    where d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    References:
        - https://arxiv.org/pdf/2302.07966.pdf
    """

    _num_qudits = 1

    def __init__(self, radix: int = 3) -> None:
        """
        Construct a ClockGate.

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
        diags = [np.exp(2j * np.pi * i / radix) for i in range(radix)]
        self._utry = UnitaryMatrix(np.diag(diags), self.radixes)

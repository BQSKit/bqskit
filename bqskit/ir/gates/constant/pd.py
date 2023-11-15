"""This module implements the PDGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class PDGate(ConstantGate, QuditGate):
    """
    The one-qudit P[i] gate.

    The clock gate is given by the following formula:

    .. math::
        P[i]_d = \\sum_{j} (-\\omega^2)^{\\delta_{ij}} |j\\rangle\\langle j|

    where

    .. math::
        \\omega = \\exp(\\frac{2\\pi i}{d})

    and d is the number of levels (2 levels is a qubit,
    3 levels is a qutrit, etc.)

    References:
        - https://pubs.aip.org/aip/jmp/article-abstract/56/3/032202/763827
    """

    _num_qudits = 1
    _num_params = 0

    def __init__(self, index: int, radix: int = 3) -> None:
        """
        Construct a PDGate.

        Args:
            index (int): The level for the phase qudit gate (<radix).

            radix (int): The number of qudit levels (>=2). Defaults to
                qutrits.

        Raises:
            ValueError: if radix < 2

            ValueError: if index >= radix
        """
        if not is_integer(radix):
            raise TypeError(f'Expected integer for radix, got {type(radix)}.')

        if radix < 2:
            raise ValueError(f'Radix must be greater than 1, got {radix}.')

        if not is_integer(index):
            raise TypeError(f'Expected integer for index, got {type(index)}.')

        if index >= radix:
            raise ValueError(
                'PDGate index must be less than number of levels:'
                f'got: {index=} >= {radix=}.',
            )

        self._radix = radix
        self.index = index

        # Calculate unitary
        omega = np.exp(2j * np.pi * index / radix)
        diags = [(-omega ** 2) if i == index else 1 for i in range(radix)]
        self._utry = UnitaryMatrix(np.diag(diags), self.radixes)

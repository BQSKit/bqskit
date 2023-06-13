"""This module implements the CSUMGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CSUMGate(ConstantGate, QutritGate):
    """
    The two-qutrit Conditional-SUM gate.

    The CSUM gate is given by the following unitary:
    """

    _num_qudits = 2
    vec = {
        0: np.array([1, 0, 0]), 1: np.array(
        [0, 1, 0],
        ), 2: np.array([0, 0, 1]),
    }
    result = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            result += np.outer(
                np.kron(vec[i], vec[j]),
                np.kron(vec[i], vec[(i + j) % 3]),
            )

    _utry = UnitaryMatrix(result.tolist())

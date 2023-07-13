"""This module implements the SdgGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SdgGate(ConstantGate, QutritGate):
    """
    The single-qutrit S Dagger gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 \\\\
        0 & 1 & 0 \\\\
        0 & 0 & w^2
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'sdg'
    _w = np.exp(2 * np.pi * 1j / 3)
    _utry = UnitaryMatrix(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, _w**2],
        ], [3],
    )

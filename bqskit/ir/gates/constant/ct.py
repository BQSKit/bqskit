"""This module implements the CTGate."""
from __future__ import annotations

import cmath

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CTGate(ConstantGate, QubitGate):
    """
    The Controlled-T gate.

    The CT gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & \\exp({i\\frac{\\pi}{4}}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'ct'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, cmath.exp(1j * cmath.pi / 4)],
        ],
    )

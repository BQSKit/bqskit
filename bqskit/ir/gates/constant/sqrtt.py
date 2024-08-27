"""This module implements the SqrtTGate."""
from __future__ import annotations

import cmath

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtTGate(ConstantGate, QubitGate):
    """
    The single-qubit square root T gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i\\frac{\\pi}{8}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'st'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, cmath.exp(1j * cmath.pi / 8)],
        ],
    )

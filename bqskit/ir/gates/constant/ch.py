"""This module implements the CHGate."""
from __future__ import annotations

import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CHGate(ConstantGate, QubitGate):
    """
    The controlled-Hadamard gate.

    The Controlled-H gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} & \\frac{\\sqrt{2}}{2} \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'ch'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2],
            [0, 0, math.sqrt(2) / 2, -math.sqrt(2) / 2],
        ],
    )

"""This module implements the XXGate."""
from __future__ import annotations

import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class XXGate(ConstantGate, QubitGate):
    """
    The Ising XX coupling gate.

    The XX gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & 0 & 0 & -\\frac{\\sqrt{2}}{2}i \\\\
        0 & \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2}i & 0 \\\\
        0 & -\\frac{\\sqrt{2}}{2}i & \\frac{\\sqrt{2}}{2} & 0 \\\\
        -\\frac{\\sqrt{2}}{2}i & 0 & 0 & \\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'rxx(pi/2)'
    _utry = UnitaryMatrix(
        [
            [math.sqrt(2) / 2, 0, 0, -1j * math.sqrt(2) / 2],
            [0, math.sqrt(2) / 2, -1j * math.sqrt(2) / 2, 0],
            [0, -1j * math.sqrt(2) / 2, math.sqrt(2) / 2, 0],
            [-1j * math.sqrt(2) / 2, 0, 0, math.sqrt(2) / 2],
        ],
    )

"""This module implements the SycamoreGate."""
from __future__ import annotations

import cmath
import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SycamoreGate(ConstantGate, QubitGate):
    """
    The SycamoreGate gate.

    The Sycamore gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & -i & 0 \\\\
        0 & -i & 0 & 0 \\\\
        0 & 0 & 0 & e^{-i\frac{\\pi}{6}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'syc'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, cmath.exp(-1j * math.pi / 6)],
        ],
    )

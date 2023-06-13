"""This module implements the SqrtISwapGate."""
from __future__ import annotations

import math

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtISwapGate(ConstantGate, QubitGate):
    """
    The square root two qubit swap and phase iSWAP gate.

    The SqrtISwap gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\frac{1}{\\sqrt{2}} & \\frac{i}{\\sqrt{2}} & 0 \\\\
        0 & \\frac{i}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'sqisw'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1 / math.sqrt(2), 1j / math.sqrt(2), 0],
            [0, 1j / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
    )

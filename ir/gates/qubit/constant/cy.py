"""This module implements the CYGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CYGate(ConstantGate, QubitGate):
    """
    The Controlled-Y gate.

    The CY gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & -i \\\\
        0 & 0 & i & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'cy'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ],
    )

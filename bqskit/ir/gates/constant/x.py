"""This module implements the XGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class XGate(ConstantGate, QubitGate):
    """
    The Pauli X gate.

    The X gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & 1 \\\\
        1 & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'x'
    _utry = UnitaryMatrix(
        [
            [0, 1],
            [1, 0],
        ],
    )

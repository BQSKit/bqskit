"""This module implements the ZGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ZGate(ConstantGate, QubitGate):
    """
    The Pauli Z gate.

    The Z gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'z'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, -1],
        ],
    )

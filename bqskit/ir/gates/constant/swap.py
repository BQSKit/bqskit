"""This module implements the SwapGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SwapGate(ConstantGate, QubitGate):
    """
    The two-qubit swap gate.

    The swap gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'swap'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
    )

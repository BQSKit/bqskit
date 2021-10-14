"""This module implements the SdgGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SdgGate(ConstantGate, QubitGate):
    """
    The single-qubit S Dagger gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & -i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'sdg'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, -1j],
        ],
    )

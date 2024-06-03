"""This module implements the SGate."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SGate(ConstantGate, QubitGate):
    """
    The single-qubit S gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 's'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, 1j],
        ],
    )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.sdg import SdgGate
        return SdgGate()

"""This module implements the TGate."""
from __future__ import annotations

import cmath

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TGate(ConstantGate, QubitGate):
    """
    The single-qubit T gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i\\frac{\\pi}{4}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 't'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, cmath.exp(1j * cmath.pi / 4)],
        ],
    )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.tdg import TdgGate
        return TdgGate()

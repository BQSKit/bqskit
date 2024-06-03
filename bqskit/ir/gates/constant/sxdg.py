"""This module implements the SqrtXDGGate/SXDGGate."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtXDGGate(ConstantGate, QubitGate):
    """
    The Sqrt(X) Dagger gate.

    The SX Dagger gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{1}{2} - \\frac{1}{2}i & \\frac{1}{2} + \\frac{1}{2}i \\\\
        \\frac{1}{2} + \\frac{1}{2}i & \\frac{1}{2} - \\frac{1}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'sxdg'
    _utry = UnitaryMatrix(
        [
            [0.5 - 0.5j, 0.5 + 0.5j],
            [0.5 + 0.5j, 0.5 - 0.5j],
        ],
    )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.sx import SXGate
        return SXGate()


SXDGGate = SqrtXDGGate

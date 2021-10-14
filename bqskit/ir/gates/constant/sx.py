"""This module implements the SqrtXGate/SXGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtXGate(ConstantGate, QubitGate):
    """
    The Sqrt(X) gate.

    The SX gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{1}{2} + \\frac{1}{2}i & \\frac{1}{2} - \\frac{1}{2}i \\\\
        \\frac{1}{2} - \\frac{1}{2}i & \\frac{1}{2} + \\frac{1}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'sx'
    _utry = UnitaryMatrix(
        [
            [0.5 + 0.5j, 0.5 - 0.5j],
            [0.5 - 0.5j, 0.5 + 0.5j],
        ],
    )


SXGate = SqrtXGate

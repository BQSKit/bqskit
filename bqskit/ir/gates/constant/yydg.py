"""This module implements the YYDGGate."""
from __future__ import annotations

import math

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class YYDGGate(ConstantGate, QubitGate):
    """
    The Ising YY Dagger coupling gate.

    The YY Dagger gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & 0 & 0 & -\\frac{\\sqrt{2}}{2}i \\\\
        0 & \\frac{\\sqrt{2}}{2} & +\\frac{\\sqrt{2}}{2}i & 0 \\\\
        0 & \\frac{\\sqrt{2}}{2}i & \\frac{\\sqrt{2}}{2} & 0 \\\\
        -\\frac{\\sqrt{2}}{2}i & 0 & 0 & \\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'ryy(-pi/2)'
    _utry = UnitaryMatrix(
        [
            [math.sqrt(2) / 2, 0, 0, -1j * math.sqrt(2) / 2],
            [0, math.sqrt(2) / 2, 1j * math.sqrt(2) / 2, 0],
            [0, 1j * math.sqrt(2) / 2, math.sqrt(2) / 2, 0],
            [-1j * math.sqrt(2) / 2, 0, 0, math.sqrt(2) / 2],
        ],
    )

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        from bqskit.ir.gates.constant.yy import YYGate
        return YYGate()

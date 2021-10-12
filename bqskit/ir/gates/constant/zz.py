"""This module implements the ZZGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ZZGate(ConstantGate, QubitGate):
    """
    The Ising ZZ coupling gate.

    The ZZ gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} - \\frac{\\sqrt{2}}{2}i & 0 & 0 & 0 \\\\
        0 & \\frac{\\sqrt{2}}{2} + \\frac{\\sqrt{2}}{2}i & 0 & 0 \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} + \\frac{\\sqrt{2}}{2}i & 0 \\\\
        0 & 0 & 0 & \\frac{\\sqrt{2}}{2} - \\frac{\\sqrt{2}}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'rzz(pi/2)'
    _utry = UnitaryMatrix(
        [
            [np.sqrt(2) / 2 - 1j * np.sqrt(2) / 2, 0, 0, 0],
            [0, np.sqrt(2) / 2 + 1j * np.sqrt(2) / 2, 0, 0],
            [0, 0, np.sqrt(2) / 2 + 1j * np.sqrt(2) / 2, 0],
            [0, 0, 0, np.sqrt(2) / 2 - 1j * np.sqrt(2) / 2],
        ],
    )

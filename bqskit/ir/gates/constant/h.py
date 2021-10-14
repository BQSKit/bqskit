"""This module implements the HGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class HGate(ConstantGate, QubitGate):
    """
    The Hadamard gate.

    The H gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{\\sqrt{2}}{2} & \\frac{\\sqrt{2}}{2} \\\\
        \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'h'
    _utry = UnitaryMatrix(
        [
            [np.sqrt(2) / 2, np.sqrt(2) / 2],
            [np.sqrt(2) / 2, -np.sqrt(2) / 2],
        ],
    )

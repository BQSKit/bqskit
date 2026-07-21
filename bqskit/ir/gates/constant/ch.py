"""This module implements the CHGate."""
from __future__ import annotations

import math

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import HGate as _HGate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CHGate(Gate, CachedClass):
    """
    The controlled-Hadamard gate.

    The Controlled-H gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} & \\frac{\\sqrt{2}}{2} \\\\
        0 & 0 & \\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'ch'
    _expr = _Controlled(_HGate())
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2],
            [0, 0, math.sqrt(2) / 2, -math.sqrt(2) / 2],
        ],
    )

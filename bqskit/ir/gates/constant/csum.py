"""This module implements the CSUMGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class CSUMGate(ConstantGate, QutritGate):
    """
    The two-qutrit Conditional-SUM gate.
    Note that this is equivalent to `CSUMDGate(num_levels=3).`

    The CSUM gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    )

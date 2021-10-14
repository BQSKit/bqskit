"""This module implements the TdgGate."""
from __future__ import annotations

import cmath

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TdgGate(ConstantGate, QubitGate):
    """
    The single-qubit T Dagger gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{-i\\frac{\\pi}{4}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'tdg'
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, cmath.exp(-1j * cmath.pi / 4)],
        ],
    )

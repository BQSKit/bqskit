"""This module implements the TGate."""
from __future__ import annotations

from openqudit.expressions import TGate as _TGate

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
    _expr = _TGate()
    _utry = UnitaryMatrix(_expr())

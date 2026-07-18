"""This module implements the SqrtXdgGate/SXdgGate."""
from __future__ import annotations

from openqudit.expressions import Dagger as _Dagger
from openqudit.expressions import SXGate as _SXGate

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SqrtXdgGate(ConstantGate, QubitGate):
    """
    The Dagger(Sqrt(X)) gate.

    The SXdg gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        \\frac{1}{2} - \\frac{1}{2}i & \\frac{1}{2} + \\frac{1}{2}i \\\\
        \\frac{1}{2} + \\frac{1}{2}i & \\frac{1}{2} - \\frac{1}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'sxdg'
    _expr = _Dagger(_SXGate())
    _utry = UnitaryMatrix(_expr())


SXdgGate = SqrtXdgGate

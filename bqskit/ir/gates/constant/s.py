"""This module implements the SGate."""
from __future__ import annotations

from openqudit.expressions import SGate as _SGate

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SGate(ConstantGate, QubitGate):
    """
    The single-qubit S gate.

    .. math::

        \\begin{pmatrix}
        1 & 0 \\\\
        0 & i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 's'
    # See YGate for why `_utry` stays a hand-written exact matrix instead
    # of `_expr()` (ULP-level residuals from the general qudit formula
    # compound significantly in deep circuits).
    _expr = _SGate()
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, 1j],
        ],
    )

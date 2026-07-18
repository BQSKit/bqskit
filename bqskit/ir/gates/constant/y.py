"""This module implements the YGate."""
from __future__ import annotations

from openqudit.expressions import YGate as _YGate

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class YGate(ConstantGate, QubitGate):
    """
    The Pauli Y gate.

    The Y gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & -i \\\\
        i & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'y'
    # `_expr` powers name/num_params (radixes come from QubitGate above).
    # `_utry` stays a hand-written exact matrix rather than `_expr()`:
    # openqudit's general qudit-Y formula evaluates a complex exponential
    # for the d=2 case instead of using the exact literal, leaving a
    # ~1e-16 residual that, while individually tiny, was found to compound
    # into ~1e-7-level errors after ~100s of gate applications in deep
    # circuits (see e.g. tests/passes/rules/test_cnot2cz.py).
    _expr = _YGate()
    _utry = UnitaryMatrix(
        [
            [0, -1j],
            [1j, 0],
        ],
    )

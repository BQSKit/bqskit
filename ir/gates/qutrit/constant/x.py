"""This module implements the XGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class XGate(ConstantGate, QutritGate):
    """
    The Pauli X gate.

    The X gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & 1 & 0 \\\\
        0 & 0 & 1 \\\\
        1 & 0 & 0 \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'x'
    _utry = UnitaryMatrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], [3],
    )


class X01Gate(ConstantGate, QutritGate):
    """
    The Pauli X01 gate.

    The X01 gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & 1 & 0 \\\\
        1 & 0 & 0 \\\\
        0 & 0 & 1 \\\\
        \\end{pmatrix}
    """
    _num_qudits = 1
    _qasm_name = 'x01'
    _utry = UnitaryMatrix(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], [3],
    )


class X02Gate(ConstantGate, QutritGate):
    """
    The Pauli X02 gate.

    The X02 gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        0 & 0 & 1 \\\\
        0 & 1 & 0 \\\\
        1 & 0 & 0 \\\\
        \\end{pmatrix}
    """
    _num_qudits = 1
    _qasm_name = 'x02'
    _utry = UnitaryMatrix(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], [3],
    )


class X12Gate(ConstantGate, QutritGate):
    """
    The Pauli X02 gate.

    The X02 gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 \\\\
        0 & 0 & 1 \\\\
        0 & 1 & 0 \\\\
        \\end{pmatrix}
    """
    _num_qudits = 1
    _qasm_name = 'x12'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ], [3],
    )

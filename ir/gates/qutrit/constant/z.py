"""This module implements the ZGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class Z0Gate(ConstantGate, QutritGate):
    """
    The Qutrit Z0 gate.

    The Z0 gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        -1 & 0 & 0\\\\
        0 & 1 & 0 \\\\
        0 & 0 1 \\end{pmatrix}
    """
    _num_qudits = 1
    _qasm_name = 'z0'
    _utry = UnitaryMatrix(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], [3],
    )


class Z1Gate(ConstantGate, QutritGate):
    """
    The Qutrit Z1 gate.

    The Z1 gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0\\\\
        0 & -1 & 0 \\\\
        0 & 0 1 \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'z1'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ], [3],
    )


class Z2Gate(ConstantGate, QutritGate):
    """
    The Qutrit Z2 gate.

    The Z2 gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0\\\\
        0 & 1 & 0 \\\\
        0 & 0 -1 \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'z2'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ], [3],
    )


class ZGate(ConstantGate, QutritGate):
    """
    The Qutrit Z gate.

    The Z gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0\\\\
        0 & w & 0 \\\\
        0 & 0 w^2 \\end{pmatrix}
    """

    _num_qudits = 1
    _qasm_name = 'z'
    _w = np.exp(2 * np.pi * 1j / 3)
    _utry = UnitaryMatrix(
        [
            [1, 0, 0],
            [0, _w, 0],
            [0, 0, _w**2],
        ], [3],
    )

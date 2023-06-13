"""This module implements the TdgGate."""  # TODO adapt for qutrit
from __future__ import annotations

import cmath

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TdgGate(ConstantGate, QutritGate):
    """The single-qubit T Dagger gate for qutrit."""

    _num_qudits = 1
    _qasm_name = 'tdg'
    _w = np.exp(2 * np.pi * 1j / 3)
    _utry = UnitaryMatrix(
        [
            [1, 0, 0],
            [0, _w**(-0.25), 0],
               [0, 0, _w**(-0.5)],
        ], [3],
    )

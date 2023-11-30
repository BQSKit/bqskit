"""This module implements the Reset class."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

class Reset(QubitGate, ConstantGate):
    """
    Construct a Reset.

    Args:
        reset (int): the index of the qubit that has reset.
    """

    def __init__(self):
        self._num_qudits = 1
        self._qasm_name = 'reset'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        raise RuntimeError(
            'Cannot compute unitary for a reset.',
        )



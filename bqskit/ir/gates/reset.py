"""This module implements the Reset class."""
from __future__ import annotations

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

class Reset(Gate):
    """Pseudogate to initialize the qudit to |0>."""

    def __init__(self, radix: int = 2) -> None:
        """
        Construct a Reset.

        Args:
            radix (int): the dimension of the qudit. (Default: 2)
        """
        self._num_qudits = 1
        self._qasm_name = 'reset'
        self._radixes = tuple([radix])
        self._num_params = 0

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        raise RuntimeError(
            'Cannot compute unitary for a reset.',
        )



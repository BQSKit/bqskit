"""This module implements the IdentityGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_valid_radixes


class IdentityGate(ConstantGate):
    """An Identity (No-OP) Gate."""

    def __init__(self, size: int = 1, radixes: Sequence[int] = []) -> None:
        """
        Creates an IdentityGate, defaulting to a single-qubit identity.

        Args:
            size (int) The number of qudits this gate acts on.

            radixes (Sequence[int]): The number of orthogonal
                states for each qudit. Defaults to qubits.
        """
        if size <= 0:
            raise ValueError('Expected positive integer, got %d' % size)
        if len(radixes) != 0 and not is_valid_radixes(radixes, size):
            raise TypeError('Invalid radixes.')

        self.size = size
        self.radixes = list(radixes or [2] * size)
        self.utry = UnitaryMatrix.identity(int(np.prod(self.radixes)))
        self.qasm_name = 'identity%d' % self.size

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        param_symbols = ['a%d' % i for i in range(self.size)]
        param_str = ','.join(param_symbols)
        header = 'gate identity%d %s' % (self.size, param_str)
        body_stmts = ['\tU(0,0,0) %s;' % sym for sym in param_symbols]
        body = '\n'.join(body_stmts)
        return f'{header}\n{{\n{body}\n}}\n'

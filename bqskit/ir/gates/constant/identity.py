"""This module implements the IdentityGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_valid_radixes


class IdentityGate(ConstantGate):
    """An Identity (No-OP) Gate."""

    def __init__(
        self,
        num_qudits: int = 1,
        radixes: Sequence[int] = [],
    ) -> None:
        """
        Create an IdentityGate, defaulting to a single-qubit identity.

        Args:
            num_qudits (int): The number of qudits this gate acts on.

            radixes (Sequence[int]): The number of orthogonal
                states for each qudit. Defaults to qubits.

        Raises:
            ValueError: If `num_qudits` is nonpositive.
        """
        if num_qudits <= 0:
            raise ValueError('Expected positive integer, got %d' % num_qudits)

        if len(radixes) != 0 and not is_valid_radixes(radixes, num_qudits):
            raise TypeError('Invalid radixes.')

        self._num_qudits = num_qudits
        self._radixes = tuple(radixes or [2] * num_qudits)
        self._dim = int(np.prod(self.radixes))
        self._utry = UnitaryMatrix.identity(self.dim, self.radixes)
        self._qasm_name = 'identity%d' % self.num_qudits

    def get_qasm_gate_def(self) -> str:
        """Return a qasm gate definition block for this gate."""
        param_symbols = ['a%d' % i for i in range(self.num_qudits)]
        param_str = ','.join(param_symbols)
        header = 'gate identity%d %s' % (self.num_qudits, param_str)
        body_stmts = ['\tU(0,0,0) %s;' % sym for sym in param_symbols]
        body = '\n'.join(body_stmts)
        return f'{header}\n{{\n{body}\n}}\n'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IdentityGate)
            and self.num_qudits == other.num_qudits
            and self.radixes == other.radixes
        )

    def __hash__(self) -> int:
        return hash((self.num_qudits, self.radixes))

    def __str__(self) -> str:
        if self.is_qubit_only():
            return f'IdentityGate({self.num_qudits})'
        else:
            return f'IdentityGate({self.num_qudits}, {self.radixes})'

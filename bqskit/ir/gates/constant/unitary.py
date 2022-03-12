"""This module implements the ConstantUnitaryGate."""
from __future__ import annotations

from typing import Sequence

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ConstantUnitaryGate(ConstantGate):
    """An arbitrary constant unitary operator."""

    def __init__(
        self,
        utry: UnitaryLike,
        radixes: Sequence[int] = [],
    ) -> None:
        """
        Construct a constant unitary operator.

        Args:
            utry (UnitaryLike): The operation as a unitary matrix.

            radixes (Sequence[int]): The number of orthogonal states
                for each qudit this gate will act on. Defaults to qubits.
        """
        self._utry = UnitaryMatrix(utry, radixes)
        self._num_qudits = self._utry.num_qudits
        self._radixes = self._utry.radixes

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ConstantUnitaryGate)
            and self._utry == other._utry
        )

    def __hash__(self) -> int:
        return hash(self._utry)

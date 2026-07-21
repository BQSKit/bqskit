"""This module implements the ConstantUnitaryGate."""
from __future__ import annotations

from collections.abc import Sequence

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class ConstantUnitaryGate(Gate, CachedClass):
    """An arbitrary constant unitary operator."""

    _num_params = 0

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

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return self._utry

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ConstantUnitaryGate)
            and self._utry == other._utry
        )

    def __hash__(self) -> int:
        return hash(self._utry)

"""This module implements the ConstantUnitaryGate."""
from __future__ import annotations

from typing import Sequence

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ConstantUnitaryGate(ConstantGate):
    """A constant unitary operator."""

    def __init__(
        self,
        utry: UnitaryLike,
        radixes: Sequence[int] = [],
    ) -> None:
        self._utry = UnitaryMatrix(utry, radixes)
        self._num_qudits = self._utry.num_qudits
        self._radixes = self._utry.radixes

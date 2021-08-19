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
        self.utry = UnitaryMatrix(utry, radixes)
        self.size = self.utry.size
        self.radixes = self.utry.radixes

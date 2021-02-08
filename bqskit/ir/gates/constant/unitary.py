"""This module implements the ConstantUnitaryGate."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitarymatrix import UnitaryLike, UnitaryMatrix

class ConstantUnitaryGate(ConstantGate):
    """A constant unitary operator."""

    def __init__(self, utry: UnitaryLike, radixes: Optional[Sequence[int]] = None) -> None:
        self.utry = UnitaryMatrix(utry)
        if radixes is not None:
            self.radixes = list(radixes)
        elif self.utry.is_qubit_unitary():
            self.radixes = [2] * self.utry.get_num_qubits()
        else:
            raise ValueError("Unable to determine radix of unitary gate."
                             "Please provide radixes argument.")
        self.size = len(radixes)


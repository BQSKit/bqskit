"""This module implements the Unitary Gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.fixedgate import FixedGate
from bqskit.qis.unitarymatrix import UnitaryMatrix

class FixedUnitaryGate(FixedGate):
    """A multi-qudit fixed unitary operator."""

    def __init__(self, utry: np.ndarray | UnitaryMatrix, radixes: Sequence[int]) -> None:
        self.utry = UnitaryMatrix(utry)
        self.radixes = list(radixes)


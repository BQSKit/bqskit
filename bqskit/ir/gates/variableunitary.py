from __future__ import annotations
from typing import Sequence
import numpy as np

from bqskit.ir.gate import Gate
from bqskit.qis.unitarymatrix import UnitaryMatrix

# TODO: UNCACHED
class VariableUnitaryGate(Gate):
    """A Variable n-qudit unitary operator."""

    def __init__(self, utry: np.ndarray | UnitaryMatrix) -> None:
        self.utry = UnitaryMatrix(utry)

    def get_grad(self, params):
        raise Exception()

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        return self.utry
    
    def optimize(env_matrix):
        pass
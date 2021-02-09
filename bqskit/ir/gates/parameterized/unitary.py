"""This module implements the VariableUnitaryGate."""
from __future__ import annotations
from typing import Sequence
import numpy as np

from bqskit.ir.gate import Gate
from bqskit.qis.unitarymatrix import UnitaryMatrix

class VariableUnitaryGate(Gate):
    """A Variable n-qudit unitary operator."""

    def __new__(cls) -> VariableUnitaryGate:
        """
        Replaces cached class __new__ function.
        VariableUnitaryGate needs to be uncached.
        """
        return super().__new__(cls)

    def __init__(self, utry: np.ndarray | UnitaryMatrix) -> None:
        self.utry = UnitaryMatrix(utry)

    def get_grad(self, params):
        raise Exception()

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        return self.utry
    
    def optimize(env_matrix):
        pass
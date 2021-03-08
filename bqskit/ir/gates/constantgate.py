"""
This module implements the ConstantGate base class.

A ConstantGate is one that does not change during circuit optimization.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ConstantGate(Gate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """The ConstantGate Class."""

    num_params = 0
    utry: UnitaryMatrix

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        if hasattr(self, 'utry'):
            return self.utry

        raise AttributeError(
            'Expected utry class variable for gate %s.'
            % self.__class__.__name__,
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, See Gate for more info."""
        self.check_parameters(params)
        return np.array([])

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        return []

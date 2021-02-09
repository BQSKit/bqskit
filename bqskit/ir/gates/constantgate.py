"""
This module implements the ConstantGate base class.

A ConstantGate is one that does not change during circuit optimization.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class ConstantGate(Gate):
    """The ConstantGate Class."""

    num_params = 0
    utry: UnitaryMatrix

    def get_unitary(self, params: Sequence[float] | None = None) -> UnitaryMatrix:
        if params is not None:
            raise ValueError('Constant gates do not take parameters.')

        if hasattr(self, 'utry'):
            return self.utry

        raise AttributeError(
            'Expected utry class variable for gate %s.'
            % self.__class__.__name__,
        )

    def get_grad(self, params: Optional[Sequence[float]] = None) -> np.ndarray:
        """Returns the gradient for the gate, See Gate for more info."""
        if params is not None:
            raise ValueError('Constant gates do not take parameters.')

        return []

    def optimize(self, env_matrix) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        return []
"""This module implements the TaggedGate Class."""
from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TaggedGate(
    ComposedGate,
    LocallyOptimizableUnitary,
    DifferentiableUnitary,
):
    """
    The TaggedGate Class.

    Allows a user to place a tag on a gate.
    """

    def __init__(self, gate: Gate, tag: Any) -> None:
        """Associate `tag` with `gate`."""

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        self.gate = gate
        self.tag = tag
        self.name = 'Tagged(%s:%s)' % (gate.get_name(), tag)
        self.num_params = gate.get_num_params()
        self.size = gate.get_size()
        self.radixes = gate.get_radixes()

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0:
            self.utry = gate.get_unitary()

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)
        if hasattr(self, 'utry'):
            return self.utry

        return self.gate.get_unitary(params)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """
        Returns the gradient for this gate, see Gate for more info.

        Notes:
            The derivative of the conjugate transpose of matrix is equal
            to the conjugate transpose of the derivative.
        """
        self.check_parameters(params)
        if hasattr(self, 'utry'):
            return np.array([])

        return self.gate.get_grad(params)  # type: ignore

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        if hasattr(self, 'utry'):
            return []
        self.check_env_matrix(env_matrix)
        return self.gate.optimize(env_matrix)  # type: ignore

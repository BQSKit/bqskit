from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes


class ControlledGate(
    ComposedGate, LocallyOptimizableUnitary,
    DifferentiableUnitary,
):
    def __init__(
        self,
        gate: Gate,
        num_controls: int = 1,
        radixes: Sequence[int] = [],
    ) -> None:
        """Construct a ControlledGate."""

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s.' % type(gate))

        if not is_integer(num_controls):
            raise TypeError(
                'Expected integer for num_controls, got %s.'
                % type(num_controls),
            )

        if num_controls < 1:
            raise ValueError(
                'Expected positive integer for num_controls, got %d.'
                % num_controls,
            )

        if len(radixes) != 0 and not is_valid_radixes(radixes, num_controls):
            raise TypeError('Invalid radixes.')

        self.gate = gate
        self._num_qudits = gate.num_qudits + num_controls
        self.num_controls = num_controls
        self._radixes = tuple(radixes or [2] * self.num_qudits) + gate.radixes
        self._name = '%d-Controlled(%s)' % (num_controls, gate.name)
        self._num_params = gate.num_params

        self.Ic = np.identity(2 ** num_controls)  # TODO: General radix support
        self.It = np.identity(gate.dim)
        self.OneProj = np.zeros(self.Ic.shape)
        self.OneProj[-1, -1] = 1
        self.left = np.kron((self.Ic - self.OneProj), self.It)

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0:
            U = self.gate.get_unitary()
            right = np.kron(self.OneProj, U)
            self.utry = UnitaryMatrix(self.left + right, self.radixes)

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)
        if hasattr(self, 'utry'):
            return self.utry

        # TODO: Find reference
        U = self.gate.get_unitary(params)
        right = np.kron(self.OneProj, U)
        return UnitaryMatrix(self.left + right, self.radixes)

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

        grads = self.gate.get_grad(params)  # type: ignore
        right = np.kron(self.OneProj, grads)
        return self.left + right

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        if hasattr(self, 'utry'):
            return []
        self.check_env_matrix(env_matrix)
        raise NotImplementedError()  # TODO

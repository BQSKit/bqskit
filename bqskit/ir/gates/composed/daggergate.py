"""This module implements the DaggerGate Class."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class DaggerGate(
    ComposedGate,
    LocallyOptimizableUnitary,
    DifferentiableUnitary,
):
    """
    An arbitrary inverted gate.

    The DaggerGate is a composed gate that equivalent to the
    conjugate transpose of the input gate.

    For example:
        >>> DaggerGate(TGate()).get_unitary() == TdgGate().get_unitary()
        ... True
    """

    def __init__(self, gate: Gate) -> None:
        """
        Create a gate which is the conjugate transpose of another.

        Args:
            gate (Gate): The Gate to conjugate transpose.
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))

        self.gate = gate
        self._name = 'Dagger(%s)' % gate.name
        self._num_params = gate.num_params
        self._num_qudits = gate.num_qudits
        self._radixes = gate.radixes

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0:
            self.utry = gate.get_unitary().dagger

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        return self.gate.get_unitary(params).dagger

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.

        Notes:
            The derivative of the conjugate transpose of matrix is equal
            to the conjugate transpose of the derivative.
        """
        if hasattr(self, 'utry'):
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        return np.transpose(grads.conj(), (0, 2, 1))

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return self.utry, np.array([])

        utry, grads = self.gate.get_unitary_and_grad(params)  # type: ignore
        return utry.dagger, np.transpose(grads.conj(), (0, 2, 1))

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return []
        self.check_env_matrix(env_matrix)
        return self.gate.optimize(env_matrix.conj().T)  # type: ignore

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DaggerGate)
            and self.gate == other.gate
        )

    def __hash__(self) -> int:
        return hash(self.gate)

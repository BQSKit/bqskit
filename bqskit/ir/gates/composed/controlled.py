"""This module implements the ControlledGate class."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer


class ControlledGate(
    ComposedGate,
    QubitGate,
    DifferentiableUnitary,
):
    """
    An arbitrary controlled gate.

    Given any qubit gate, ControlledGate can add control qubits.
    """

    def __init__(
        self,
        gate: Gate,
        num_controls: int = 1,
    ) -> None:
        """
        Construct a ControlledGate.

        Args:
            gate (Gate): The gate to control.

            num_controls (int): The number of controls to add.

        Raises:
            ValueError: If `num_controls` is less than 1
        """

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

        self.gate = gate
        self._num_qudits = gate.num_qudits + num_controls
        self.num_controls = num_controls
        self._name = '%d-Controlled(%s)' % (num_controls, gate.name)
        self._num_params = gate.num_params

        self.Ic = np.identity(2 ** num_controls, dtype=np.complex128)
        self.It = np.identity(gate.dim, dtype=np.complex128)
        self.OneProj = np.zeros(self.Ic.shape, dtype=np.complex128)
        self.OneProj[-1, -1] = 1
        self.left = np.kron((self.Ic - self.OneProj), self.It)

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0:
            U = self.gate.get_unitary()
            right = np.kron(self.OneProj, U)
            self.utry = UnitaryMatrix(self.left + right, self.radixes)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        U = self.gate.get_unitary(params)
        right = np.kron(self.OneProj, U)
        return UnitaryMatrix(self.left + right, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        right = np.kron(self.OneProj, grads)
        return self.left + right

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

        U, G = self.gate.get_unitary_and_grad(params)  # type: ignore

        right = np.kron(self.OneProj, U)
        utry = UnitaryMatrix(self.left + right, self.radixes)
        grads = np.kron(self.OneProj, G)
        return utry, grads

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlledGate)
            and self.gate == other.gate
            and self.num_controls == other.num_controls
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.num_controls))

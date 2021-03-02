"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that
can be applied to a circuit.
"""
from __future__ import annotations

import abc
from typing import Callable
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np

from bqskit.qis.unitary import Unitary
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_square_matrix

if TYPE_CHECKING:
    from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate


class Gate(Unitary, CachedClass):
    """Gate Base Class."""

    name: str
    qasm_name: str

    def get_name(self) -> str:
        """Returns the name of the gate, defaults to the class name."""
        if hasattr(self, 'name'):
            return self.name

        return self.__class__.__name__

    def get_qasm_name(self) -> str:
        """Returns the qasm name for this gate."""
        if not self.is_qubit_gate():
            raise AttributeError('QASM only supports qubit gates.')

        if hasattr(self, 'qasm_name'):
            return self.qasm_name

        raise AttributeError(
            'Expected qasm_name field for gate %s.'
            % self.get_name(),
        )

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        if not self.is_qubit_gate():
            raise AttributeError('QASM only supports qubit gates.')

        return ''

    @abc.abstractmethod
    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """
        Returns the gradient for the gate as a np.ndarray.

        Args:
            params (Sequence[float]): The gate parameters.

        Returns:
            (np.ndarray): The (num_params,N,N)-shaped, matrix-by-vector
                derivative of this gate at the point specified by params.

        Note:
            The gradient of a gate is defined as a matrix-by-vector derivative.
            If the UnitaryMatrix result of get_unitary has dimension NxN, then
            the shape of get_grad's return value should equal (num_params,N,N),
            where the return value's i-th element is the matrix derivative of
            the gate's unitary with respect to the i-th parameter.
        """

    def get_unitary_and_grad(
        self,
        params: Sequence[float] = [],
    ) -> Tuple[UnitaryMatrix, np.ndarray]:
        """
        Returns a tuple combining the outputs of get_unitary and get_grad.

        Note:
            Can be overridden to speed up optimization by calculating both
            at the same time.
        """
        return (self.get_unitary(params), self.get_grad(params))

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """
        Returns optimal parameters with respect to an environment matrix.

        More specifically, return the parameters that maximize the
        real component of the trace of the product between the env_matrix
        and this gate:
            argmax(Re(Trace(env_matrix @ self.get_unitary(params))))

        Args:
            env_matrix (np.ndarray): Optimize with respect to this matrix.
                Has the same dimensions as the gate's unitary.

        Returns
            (list[float]): The parameters that optimizes this gate.
        """
        raise NotImplementedError(
            'Individual optimization not implemented'
            'for %s.' % self.get_name(),
        )

    def check_env_matrix(self, env_matrix: np.ndarray) -> None:
        """Checks to ensure the env_matrix is valid and matches the gate."""
        if not is_square_matrix(env_matrix):
            raise TypeError('Expected a sqaure matrix.')

        if env_matrix.shape != (self.get_dim(), self.get_dim()):
            raise TypeError('Enviromental matrix shape mismatch.')

    with_frozen_params: Callable[[Gate, dict[int, float]], FrozenParameterGate]
    with_all_frozen_params: Callable[[Gate, list[float]], FrozenParameterGate]

    def __repr__(self) -> str:
        return self.get_name()

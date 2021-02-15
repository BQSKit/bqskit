"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that
can be applied to a circuit.
"""
from __future__ import annotations

import abc
from typing import Callable
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np

from bqskit.qis.unitary import Unitary
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_sequence, is_square_matrix

if TYPE_CHECKING:
    from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate


class Gate(Unitary, CachedClass):
    """Gate Base Class."""

    name: str
    num_params: int
    radixes: list[int]
    size: int
    qasm_name: str

    def get_name(self) -> str:
        """Returns the name of the gate, defaults to the class name."""
        if hasattr(self, 'name'):
            return self.name

        return self.__class__.__name__

    def get_num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        if hasattr(self, 'num_params'):
            return self.num_params

        raise AttributeError(
            'Expected num_params field for gate %s.'
            % self.get_name(),
        )

    def get_radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        if hasattr(self, 'radixes'):
            return self.radixes

        raise AttributeError(
            'Expected radixes field for gate %s.'
            % self.get_name(),
        )

    def get_size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        if hasattr(self, 'size'):
            return self.size

        raise AttributeError(
            'Expected size field for gate %s.'
            % self.get_name(),
        )
    
    def get_dim(self) -> int:
        """Returns the matrix dimension for this gate's unitary."""
        return int(np.prod(self.get_radixes()))

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

    def is_qubit_gate(self) -> bool:
        """Returns true if this gate only acts on qubits."""
        return all([radix == 2 for radix in self.get_radixes()])

    def is_qutrit_gate(self) -> bool:
        """Returns true if this gate only acts on qutrits."""
        return all([radix == 3 for radix in self.get_radixes()])

    def is_parameterized(self) -> bool:
        """Returns true if this gate is a parameterized gate."""
        return self.get_num_params() != 0

    def is_constant(self) -> bool:
        """Returns true if this gate doesn't change during optimization."""
        return not self.is_parameterized()

    def check_parameters(self, params: Sequence[float]) -> None:
        """Checks to ensure parameters are valid and match the gate."""
        if not is_sequence(params):
            raise TypeError(
                'Expected a sequence type for params, got %s.'
                % type(params),
            )

        if not all(isinstance(p, (float, int)) for p in params):
            typechecks = [isinstance(p, (float, int)) for p in params]
            fail_idx = typechecks.index(False)
            raise TypeError(
                'Expected params to be floats, got %s.'
                % type(params[fail_idx]),
            )

        if len(params) != self.get_num_params():
            raise ValueError(
                'Expected %d params, got %d.'
                % (self.get_num_params(), len(params)),
            )
    
    def check_env_matrix(self, env_matrix: np.ndarray) -> None:
        """Checks to ensure the env_matrix is valid and matches the gate."""
        if not is_square_matrix(env_matrix):
            raise TypeError("Expected a sqaure matrix.")

        if env_matrix.shape != (self.get_dim(), self.get_dim()):
            raise TypeError("Enviromental matrix shape mismatch.")

    with_frozen_params: Callable[[Gate, dict[int, float]], FrozenParameterGate]
    with_all_frozen_params: Callable[[Gate, list[float]], FrozenParameterGate]

    def __repr__(self) -> str:
        return self.get_name()

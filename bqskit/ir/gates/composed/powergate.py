"""This module implements the DaggerGate Class."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.utils.typing import is_integer
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs


class PowerGate(
    ComposedGate,
    LocallyOptimizableUnitary,
    DifferentiableUnitary,
):
    """
    An arbitrary inverted gate.

    The PowerGate is a composed gate that equivalent to the
    integer power of the input gate.

    For example:
        >>> from bqskit.ir.gates import TGate, TdgGate
        >>> PowerGate(TGate(),2).get_unitary() == TdgGate().get_unitary()*TdgGate().get_unitary()
        True
    """

    def __init__(self, gate: Gate, power: int = 1) -> None:
        """
        Create a gate which is the integer power of the input gate.

        Args:
            gate     (Gate): The Gate to conjugate transpose.
            power (integer): The power index for the PowerGate 
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s' % type(gate))
        
        if not is_integer(power):
            raise TypeError(
                f'Expected integer for num_controls, got {type(power)}.',)
        
        self.gate = gate
        self.power =power
        self._name = 'Power(%s)' % gate.name
        self._num_params = gate.num_params
        self._num_qudits = gate.num_qudits
        self._radixes = gate.radixes

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            self.utry = np.linalg.matrix_power(gate.get_unitary(),self.power)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        return np.linalg.matrix_power(self.gate.get_unitary(params),power)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.

        Notes:
            The derivative of the integer power of matrix is equal
            to the derivative of the matrix multiplied by the integer-1 power of the matrix
            and by the integer power.
        """
        if hasattr(self, 'utry'):
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        return self.power*np.linalg.matrix_power(self.gate.get_unitary(params),power-1)@grads
    

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
        return np.linalg.matrix_power(utry,power), self.power*np.linalg.matrix_power(utry,self.power-1)@grads
    
    
    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]: #TODO fix
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
            isinstance(other, PowerGate)
            and self.gate == other.gate
        )

    def __hash__(self) -> int:
        return hash(self.gate)

    def get_inverse(self) -> Gate:
        """Return the gate's inverse as a gate."""
        return self.gate

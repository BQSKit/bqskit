"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation
that can be applied to a circuit.
"""

from __future__ import annotations
import abc
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from typing import Optional, Sequence

import numpy as np

from bqskit.qis.unitary import Unitary
from bqskit.utils.cachedclass import CachedClass


class Gate(Unitary, CachedClass):
    """Gate Base Class."""

    name: str
    num_params: int
    radixes: list[int]
    size: int
    qasm_name: str

    def get_name(self) -> str:
        """Returns the name of the gate, defaults to the class name."""
        if hasattr( self, "name" ):
            return self.name

        return self.__class__.__name__

    def get_num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        if hasattr( self, "num_params" ):
            return self.num_params

        raise AttributeError(
            'Expected num_params field for gate %s.'
            % self.get_name(),
        )

    def get_radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        if hasattr( self, "radixes" ):
            return self.radixes

        raise AttributeError(
            'Expected radixes field for gate %s.'
            % self.get_name(),
        )

    def get_size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        if hasattr( self, "size" ):
            return self.size

        raise AttributeError(
            'Expected size field for gate %s.'
            % self.get_name(),
        )
    
    def get_qasm_name(self) -> str:
        """Returns the qasm name for this gate."""
        if hasattr( self, "qasm_name" ):
            return self.qasm_name

        raise AttributeError(
            'Expected qasm_name field for gate %s.'
            % self.get_name(),
        )
    
    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return ""
    
    @abc.abstractmethod
    def get_grad(self, params: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        Returns the gradient for the gate as a np.ndarray.

        Args:
            params (Optional[Sequence[float]]): The gate parameters.
        
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
    
    @abc.abstractmethod
    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
    
    def is_qubit_gate(self) -> bool:
        """Returns true if this gate only acts on qubits."""
        return all( [ radix == 2 for radix in self.get_radixes() ] )
    
    def is_qutrit_gate(self) -> bool:
        """Returns true if this gate only acts on qutrits."""
        return all( [ radix == 3 for radix in self.get_radixes() ] )
    
    def is_parameterized(self) -> bool:
        """Returns true if this gate is a parameterized gate."""
        return self.get_num_params() != 0
    
    def is_constant(self) -> bool:
        """Returns true if this gate doesn't change during optimization."""
        return not self.is_parameterized()

    def __repr__(self) -> str:
        return self.name

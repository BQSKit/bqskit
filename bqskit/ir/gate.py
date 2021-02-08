"""
This module implements the Gate base class.

A gate is a potentially parameterized unitary operation
that can be applied to a circuit.
"""
from __future__ import annotations
import abc
from typing import Optional, Sequence

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

        if hasattr( self.__class__, "name" ):
            return self.__class__.name

        return self.__class__.__name__

    def get_num_params(self) -> int:
        """Returns the number of parameters for this gate."""
        if hasattr( self, "num_params" ):
            return self.num_params
        
        if hasattr( self.__class__, "num_params" ):
            return self.__class__.num_params

        raise AttributeError(
            'Expected num_params field for gate %s.'
            % self.get_name(),
        )

    def get_radixes(self) -> list[int]:
        """Returns the number of orthogonal states for each qudit."""
        if hasattr( self, "radixes" ):
            return self.radixes
        
        if hasattr( self.__class__, "radixes" ):
            return self.__class__.radixes

        raise AttributeError(
            'Expected radixes field for gate %s.'
            % self.get_name(),
        )

    def get_size(self) -> int:
        """Returns the number of qudits this gate acts on."""
        if hasattr( self, "size" ):
            return self.size

        if hasattr( self.__class__, "size" ):
            return self.__class__.size

        raise AttributeError(
            'Expected size field for gate %s.'
            % self.get_name(),
        )
    
    def get_qasm_name(self) -> str:
        """Returns the qasm name for this gate."""
        if hasattr( self, "qasm_name" ):
            return self.qasm_name

        if hasattr( self.__class__, "qasm_name" ):
            return self.__class__.qasm_name

        raise AttributeError(
            'Expected qasm_name field for gate %s.'
            % self.get_name(),
        )
    
    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        return ""
    
    @abc.abstractmethod
    def get_grad(self, params: Optional[Sequence[float]] = None) -> list[float]:
        """Returns the gradient for the gate as a list of floats."""
    
    @abc.abstractmethod
    def optimize(self, env_matrix) -> None:
        """Optimizes the gate with respect to an environment matrix."""
    
    def is_qubit_gate(self) -> bool:
        """Returns true if this gate only acts on qubits."""
        return all( [ radix == 2 for radix in self.get_radixes() ] )
    
    def is_qutrit_gate(self) -> bool:
        """Returns true if this gate only acts on qutrits."""
        return all( [ radix == 3 for radix in self.get_radixes() ] )

    def __repr__(self) -> str:
        return self.name

"""
This module implements the Circuit class.

A circuit represents a quantum program composed of gate objects.
"""
from __future__ import annotations
from bqskit.utils.typing import is_iterable, is_sequence, is_valid_location
import logging

from typing import Iterable, Optional
from typing import Iterator
from typing import Sequence

import numpy as np

from bqskit.ir.cell import CircuitCell
from bqskit.ir.gate import Gate
from bqskit.qis.unitary import Unitary

_logger = logging.getLogger( __name__ )

class Circuit(Unitary):
    """The Circuit class."""

    def __init__(
        self, num_qudits: int,
        qudit_radixes: Optional[list[int]] = None,
    ) -> None:
        """
        Circuit constructor. Builds an empty circuit with
        the specified number of qudits. By default, all qudits
        are qubits, but this can be changed with qudit_radixes parameter.

        Args:
            num_qudits (int): The number of qudits in this circuit.

            qudit_radixes (List[int]): A list with length equal
                to num_qudits. Each element specifies the base
                of a qudit. Defaults to qubits.

        Raises:
            ValueError: if num_qudits is non-positive.

        Examples:
            circ = Circuit(4) # Creates four-qubit empty circuit.
        """

        if not isinstance(num_qudits, int):
            raise TypeError(
                'Invalid type for num_qudits: '
                'expected int, got %s.' % type(num_qudits),
            )
        if num_qudits <= 0:
            raise ValueError('Expected positive number for num_qudits.')

        self.num_qudits = num_qudits
        self.qudit_radixes = qudit_radixes or [2] * num_qudits
        self._circuit: list[list[CircuitCell]] = []
        self.gate_set: list[Gate] = []

    @property
    def num_params(self) -> int:
        """The total number of parameters in the circuit."""
        pass

    @property
    def num_gates(self) -> int:
        """The total number of gates in the circuit."""
        pass

    @property
    def time_steps(self) -> int:
        """The number of time steps in the circuit."""
        return len(self._circuit)
    
    @property
    def depth(self) -> int:
        """The length of the critical path in the circuit."""
        pass

    def get_gate(self, qudit: int, time_step: int) -> Gate:
        pass

    def append_gate(self, gate: Gate, location: Iterable[int], params: Optional[Sequence[float]] = None) -> None:
        """Apply the gate object to the qudits described by location."""
        if not is_valid_location( location, self.num_qudits ):
            raise TypeError( "Invalid location." )

        if not isinstance( gate, Gate ):
            raise TypeError( "Invalid gate." )

        if len( location ) != gate.get_gate_size():
            raise ValueError( "Gate and location size mismatch." )

        for gate_radix, circ_radix_idx in zip( gate.get_radix(), location ):
            if gate_radix != self.qudit_radixes[circ_radix_idx]:
                raise ValueError( "Gate and location radix mismatch." )
        
        if params is not None and len( params ) != gate.get_num_params():
            raise ValueError( "Gate and parameter mismatch.")
        elif params is None:
            params = [0.0] * gate.get_num_params()  # TODO: Re-evaluate later

        try:
            gate_index = self.gate_set.index(gate)
        except ValueError:
            self.gate_set.append(gate)
            gate_index = len( self.gate_set )
        
        cell = CircuitCell(gate_index, location, params)

        time_step = self.find_available_time_step( location )

        if time_step == -1:
            self.append_time_step()
        
        for qudit_index in location:
            self._circuit[time_step][qudit_index] = cell
    
    def find_available_time_step(self, location: Iterable[int]) -> int:
        """
        Finds the first available time step where all qudits described
        in location are free.
        """
        if not is_valid_location( location, self.num_qudits ):
            raise TypeError( "Invalid location." )

        for time_step in range( self.time_step ):
            if self.is_time_step_available( time_step, location ):
                return time_step

        return -1
    
    def is_time_step_available(self, time_step: int, location: Iterable[int]) -> bool:
        """Checks if the time_step has all qudits in location available."""
        if not is_valid_location( location, self.num_qudits ):
            raise TypeError( "Invalid location." )

        for qudit_index in location:
            if self._circuit[time_step][qudit_index] is not None:
                return False
            
        return True
    
    def __getitem__( self, position: Sequence[int] ) -> CircuitCell:
        if not is_sequence( position ):
            raise TypeError("Invalid position.")

        if len( position ) != 2:
            raise ValueError("Invalid dimensions in position.")

        if is_iterable( position[0] ) or is_iterable( position[1] ):
            raise TypeError("Currently, we do not support slicing.")
        
        return self._circuit[position[0]][position[1]]

    def remove_gate(self, qudit: int, time_step: int) -> None:
        pass

    def insert_gate(
        self, gate: Gate, qudits: Iterable[int],
        time_step: int,
    ) -> None:
        pass

    def get_unitary(self, params: list[float] | None = None) -> np.ndarray:
        assert(params is None or len(params) == self.get_num_params())

    def __iter__(self) -> Iterator[Gate]:
        pass

    def __str__(self) -> str:
        pass

    def __add__(self, rhs: Circuit) -> Circuit:
        pass

    def __mul__(self, rhs: int) -> Circuit:
        pass

    def save(self, filename: str) -> None:
        pass

    @staticmethod
    def from_file(filename: str) -> Circuit:
        pass

    @staticmethod
    def from_str(str: str) -> Circuit:
        pass

    @staticmethod
    def from_unitary(utry: np.ndarray) -> Circuit:
        pass

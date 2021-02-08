"""
This module implements the Circuit class.

A circuit represents a quantum program composed of gate objects.
"""
from __future__ import annotations

import logging
from typing import Iterable
from typing import Iterator
from typing import Sequence

import numpy as np

from bqskit.ir.cell import CircuitCell
from bqskit.ir.gate import Gate
from bqskit.qis.unitary import Unitary
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_valid_location
from bqskit.utils.typing import is_valid_radixes


_logger = logging.getLogger(__name__)


class Circuit(Unitary):
    """The Circuit class."""

    def __init__(
        self, num_qudits: int,
        qudit_radixes: Sequence[int] | None = None,
    ) -> None:
        """
        Circuit constructor. Builds an empty circuit with
        the specified number of qudits. By default, all qudits
        are qubits, but this can be changed with qudit_radixes.

        Args:
            num_qudits (int): The number of qudits in this circuit.

            qudit_radixes (Optional[Sequence[int]]): A sequence with its
                length equal to num_qudits. Each element specifies the
                base of a qudit. Defaults to qubits.

        Raises:
            ValueError: if num_qudits is nonpositive.

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

        if not is_valid_radixes(self.qudit_radixes, self.num_qudits):
            raise TypeError('Invalid qudit radixes.')

        self._circuit: list[list[CircuitCell | None]] = []
        self.gate_set: dict[Gate, int] = {}

    @property
    def num_params(self) -> int:
        """The total number of parameters in the circuit."""
        num_params_acm = 0
        for gate, count in self.gate_set.items():
            num_params_acm += gate.get_num_params() * count
        return num_params_acm

    @property
    def num_gates(self) -> int:
        """The total number of gates in the circuit."""
        num_gates_acm = 0
        for gate, count in self.gate_set.items():
            num_gates_acm += count
        return num_gates_acm

    @property
    def time_steps(self) -> int:
        """The number of time steps in the circuit."""
        return len(self._circuit)

    @property
    def parallelism(self) -> float:
        """The amount of parallelism in the circuit."""
        return self.num_gates / self.time_steps

    @property
    def depth(self) -> int:
        """The length of the critical path in the circuit."""

    def get_gate(self, time_step: int, qudit_index: int) -> Gate | None:
        """
        Retrieves the gate at the specified position from the circuit.
        If a gate exists at the position, this reduces to:
            self[time_step, qudit_index].gate

        Args:
            time_step (int): The time_step coordinate.

            qudit_index (int): The qudit coordinate.

        Returns:
            (Gate | None): The gate at the specified position or None.

        Raises:
            IndexError: If the specified position doesn't exist.

        Examples:
            >>> circ = Circuit(2)
            >>> circ.append_gate(gates.H, [0])
            >>> circ.append_gate(gates.X, [0])
            >>> circ.append_gate(gates.Z, [1])
            >>> circ.get_gate(0, 0)
            gates.H
            >>> circ.get_gate(1, 0)
            gates.X
            >>> circ.get_gate(0, 1)
            gates.Z
            >>> circ.get_gate(1, 1)
            None
        """

        cell = self[time_step, qudit_index]

        if cell is None:
            return None

        return cell.gate

    def append_gate(self, gate: Gate, location: Iterable[int], params: Sequence[float] | None = None) -> None:
        """
        Append the gate object to the circuit on the qudits described
        by location. Optionally, you can specify parameters for the gate.
        By default, the params are zeroed.

        Args:
            gate (Gate): The gate to append.

            location (Iterable[int]): Apply the gate to this set of qudits.

            params (Optional[Sequence[float]]): The gate's parameters.

        Examples:
            >>> circ = Circuit(1)
            >>> # Append a Hadamard gate to qudit 0.
            >>> circ.append_gate(gates.H, [0])
        """

        if not isinstance(gate, Gate):
            raise TypeError('Invalid gate.')

        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        if len(location) != gate.size:
            raise ValueError('Gate and location size mismatch.')

        for gate_radix, circ_radix_idx in zip(gate.radixes, location):
            if gate_radix != self.qudit_radixes[circ_radix_idx]:
                raise ValueError('Gate and location radix mismatch.')

        if params is None:
            params = [0.0] * gate.num_params  # TODO: Re-evaluate later

        if len(params) != gate.num_params:
            raise ValueError('Gate and parameter mismatch.')

        if gate not in self.gate_set:
            self.gate_set[gate] = 0
        self.gate_set[gate] += 1

        cell = CircuitCell(gate, location, params)

        time_step = self.find_available_time_step(location)

        if time_step == -1:
            self.append_time_step()

        for qudit_index in location:
            self._circuit[time_step][qudit_index] = cell

    def insert_gate(self, gate: Gate, time_step: int, location: Iterable[int], params: Sequence[float] | None = None) -> None:
        """
        Insert the gate object in the circuit on the qudits described
        by location at the time_step specified. Optionally, you can
        specify parameters for the gate. By default, the params are zeroed.
        After this, you can expect:
            all( [ self.get_gate(time_step, idx) == gate
                   for idx in location ] )

        Args:
            gate (Gate): The gate to insert.

            time_step (int): The time_step to insert the gate.

            location (Iterable[int]): Apply the gate to this set of qudits.

            params (Optional[Sequence[float]]): The gate's parameters.

        Raises:
            IndexError: If the specified time_step doesn't exist.

        Examples:
            >>> circ = Circuit(1)
            >>> # Append a Hadamard gate to qudit 0.
            >>> circ.append_gate(gates.H, [0])
            >>> # Insert a X gate at the beginning
            >>> circ.insert_gate(gates.X, 0, [0])
            >>> circ.get_Gate(0, 0)
            gates.X
            >>> circ.get_Gate(1, 0)
            gates.H

        Notes:
            Supports negative indexing.
        """

        if not isinstance(gate, Gate):
            raise TypeError('Invalid gate.')

        if time_step < -self.time_steps or time_step >= self.time_steps:
            raise IndexError('Invalid time_step.')

        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        if len(location) != gate.size:
            raise ValueError('Gate and location size mismatch.')

        for gate_radix, circ_radix_idx in zip(gate.radixes, location):
            if gate_radix != self.qudit_radixes[circ_radix_idx]:
                raise ValueError('Gate and location radix mismatch.')

        if params is None:
            params = [0.0] * gate.num_params  # TODO: Re-evaluate later

        if len(params) != gate.num_params:
            raise ValueError('Gate and parameter mismatch.')

        if time_step < 0:
            time_step = self.time_steps - time_step

        if gate not in self.gate_set:
            self.gate_set[gate] = 0
        self.gate_set[gate] += 1

        cell = CircuitCell(gate, location, params)

        if not self.is_time_step_available(time_step, location):
            self.insert_time_step(time_step)

        for qudit_index in location:
            self._circuit[time_step][qudit_index] = cell

    # TODO: Remove Empty Timesteps
    def remove_gate(self, time_step: int, qudit_index: int) -> None:
        """
        Removes the gate at the specified position if it is occupied,
            otherwise does nothing.

        Args:
            time_step (int): The time_step coordinate.

            qudit_index (int): The qudit_index coordinate.

        Raises:
            IndexError: If the specified position doesn't exist.

        Examples:
            >>> circ = Circuit(1)
            >>> # Append a Hadamard gate to qudit 0.
            >>> circ.append_gate(gates.H, [0])
            >>> circ.remove_gate(0, 0)
            >>> circ.num_gates
            0
        """

        cell = self[time_step, qudit_index]

        if cell is None:
            return

        for gate_qudit_index in cell.location:
            self._circuit[time_step, gate_qudit_index] = None

        del cell

    def get_unitary(self, params: list[float] | None = None) -> UnitaryMatrix:
        assert(params is None or len(params) == self.num_params)

    def find_available_time_step(self, location: Iterable[int]) -> int:
        """
        Finds the first available time step where all qudits described
        in location are free. Returns -1 if no suitable time_step found.

        Args:
            localtion (Iterable[int]): Find a time_step for this location.

        Examples:
            >>> circ = Circuit(2)
            >>> circ.append_gate(gates.H, [0])
            >>> circ.find_available_time_step([1])
            0
            >>> circ.find_available_time_step([0])
            -1
        """

        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        # Iterate through time_steps in reverse order
        # Find the first unavailable time_step
        # The first available time_step is the previous one
        for time_step in range(self.time_steps - 1, -1, -1):
            if not self.is_time_step_available(time_step, location):
                if time_step == self.time_steps - 1:
                    return -1
                return time_step + 1

        return -1

    def is_time_step_available(self, time_step: int, location: Iterable[int]) -> bool:
        """Checks if the time_step has all qudits in location available."""
        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        for qudit_index in location:
            if self._circuit[time_step][qudit_index] is not None:
                return False

        return True

    def append_time_step(self) -> None:
        """Appends an empty time_step to the end of the circuit."""
        self._circuit.append([None] * self.num_qudits)

    def insert_time_step(self, time_step: int) -> None:
        """Inserts an empty time_step in the circuit."""
        self._circuit.insert(time_step, [None] * self.num_qudits)

    def __getitem__(self, position: Sequence[int]) -> CircuitCell | None:
        if not is_sequence(position):
            raise TypeError('Invalid position.')

        if len(position) != 2:
            raise ValueError('Invalid dimensions in position.')

        if is_iterable(position[0]) or is_iterable(position[1]):
            raise TypeError('Currently, we do not support slicing.')

        return self._circuit[position[0]][position[1]]

    class CircuitIterator:
        def __init__(self, circuit: list[list[CircuitCell | None]]) -> None:
            self.circuit = circuit
            self.time_step = 0
            self.qudit_index = 0
            self.max_time_step = len( circuit )
            self.max_qudit_index = 0 if self.max_time_step == 0 else len( circuit[0] )
            self.qudits_to_skip = []
        
        def increment_iter(self) -> None:
            self.qudit_index += 1
            if self.qudit_index >= self.max_qudit_index:
                self.qudit_index = 0
                self.time_step += 1
            if self.time_step >= self.max_time_step:
                raise StopIteration
        
        def dereference(self) -> CircuitCell | None:
            return self.circuit[self.time_step][self.qudit_index]
        
        def __iter__(self) -> Iterator[CircuitCell]:
            return self
        
        def __next__(self) -> CircuitCell:
            while self.dereference() is None:
                self.increment_iter()
            return self.dereference()
            
    def __iter__(self) -> self.CircuitIterator:
        return self.CircuitIterator(self._circuit)

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

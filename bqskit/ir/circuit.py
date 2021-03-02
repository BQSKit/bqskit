"""This module implements the Circuit class."""

from __future__ import annotations
from bqskit.qis.unitarybuilder import UnitaryBuilder
from bqskit.qis.state import StateVector, StateVectorMap

import logging
from typing import Any, Collection, Generator, Iterable, Iterator, Optional
from typing import Sequence

import numpy as np

from bqskit.ir.operation import Operation
from bqskit.ir.gate import Gate
from bqskit.qis.unitary import Unitary
from bqskit.qis.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_valid_location
from bqskit.utils.typing import is_valid_radixes


_logger = logging.getLogger(__name__)


class Circuit(Unitary, StateVectorMap, Collection[Operation]):
    """
    Circuit class.

    A circuit represents a quantum program composed of gate objects.

    Invariants:
        1. A circuit method will never complete with an idle cycle.
            An idle cycle is one that contains no gates.
        
        2. No one logical operation will ever be pointed to from more than
            one cycle. The Operation object is a CachedClass, so it may appear
            multiple times throughout the circuit, but never as one logical
            operation across multiple cycles.
        
        3. Gate Set Stuff
    """

    def __init__(self, size: int, radixes: Sequence[int] = []) -> None:
        """
        Circuit constructor.
        
        Builds an empty circuit with the specified number of qudits.

        By default, all qudits are qubits, but this can be changed
        with radixes.

        Args:
            size (int): The number of qudits in this circuit.

            radixes (Sequence[int]): A sequence with its length equal
                to `size`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: if size is nonpositive.

        Examples:
            >>> circ = Circuit(4)  # Creates four-qubit empty circuit.
            >>> circ = Circuit(2, [2, 3])  # Creates one qubit and one qutrit.
        """
        if size <= 0:
            raise ValueError(
                'Expected positive integer for size'
                ', got %d.' % size
            )

        self.size = int(size)
        self.radixes = list(radixes or [2] * self.size)

        if not is_valid_radixes(self.radixes, self.size):
            raise TypeError('Invalid qudit radixes.')

        self._circuit: list[list[Operation | None]] = []
        self._gate_set: dict[Gate, int] = {}

    # region Circuit Properties

    def get_num_params(self) -> int:
        """Return the total number of parameters in the circuit."""
        num_params_acm = 0
        for gate, count in self._gate_set.items():
            num_params_acm += gate.get_num_params() * count
        return num_params_acm

    def get_num_operations(self) -> int:
        """Return the total number of gates/operations in the circuit."""
        num_gates_acm = 0
        for gate, count in self._gate_set.items():
            num_gates_acm += count
        return num_gates_acm

    def get_num_cycles(self) -> int:
        """Return the number of cycles in the circuit."""
        return len(self._circuit)

    def get_params(self) -> list[float]:
        """Returns the stored parameters for the circuit."""
        return np.sum([ cell.params for cell in self ])

    def get_depth(self) -> int:
        """Return the length of the critical path in the circuit."""
        qudit_depths = np.zeros(self.get_size())
        for op in self:
            new_depth = max(qudit_depths[op.location]) + 1
            qudit_depths[op.location] = new_depth
        return max(qudit_depths)

    def get_parallelism(self) -> float:
        """Calculate the amount of parallelism in the circuit."""
        return self.get_num_gates() / self.get_depth()
    
    def get_coupling_graph(self) -> list[tuple[int, int]]:
        """
        The qudit connectivity in the circuit.

        Notes:
            Multi-qudit gates require participating qudits to have
                all-to-all connectivity.
        """
        coupling_graph = set()
        for op in self:
            for q1 in op.location:
                for q2 in op.location:
                    if q1 == q2:
                        continue
                    coupling_graph.add((q1, q2))
        return coupling_graph
    
    # endregion

    # region Qudit Methods

    def append_qudit(self, radix: int) -> None:
        """
        Append a qudit to the circuit.

        Args:
            radix (int): The radix of the qudit.

        Raises:
            ValueError: If `radix` is < 2
        """

        if radix < 2:
            raise ValueError("Expected radix to be > 2, got %d" % radix)

        self.size += 1
        self.radixes.append(radix)

        for cycle in self._circuit:
            cycle.append(None)
    
    def extend_qudits(self, radixes: Sequence[int]) -> None:
        """
        Append many qudits to the circuit.

        Args:
            radixes (Sequence[int]): The radix for each qudit to append.
        
        Raises:
            ValueError: If any radix in `radixes` is < 2.
        """

        for radix in radixes:
            self.append_qudit(radix)

    def insert_qudit(self, index: int, radix: int) -> None:
        """
        Insert a qudit in to the circuit.

        Args:
            radix (int): The radix of the qudit.

            index (int): The index where to insert the qudit.

        Raises:
            ValueError: If `radix` is < 2.
        """

        if radix < 2:
            raise ValueError("Expected radix to be > 2, got %d" % radix)

        self.size += 1
        self.radixes.insert(index, radix)

        for cycle in self._circuit:
            cycle.insert(index, None)

    def pop_qudit(self, qudit_index: int) ->  None:
        """Pop a qudit from the circuit and all gates attached to it."""
        for cycle_index, cycle in enumerate(self._circuit):
            if cycle[qudit_index] is not None:
                self.pop_gate(cycle_index, qudit_index)
            cycle.pop(qudit_index)
    
    def is_qudit_in_range(self, index: int) -> bool:
        """Return true if qudit index is in-range for the circuit."""
        return index < self.get_size() and index >= -self.get_size()
    
    def is_qudit_idle(self, index: int) -> bool:
        """Return true if the qudit is idle."""
        return all(cycle[index] is None for cycle in self._circuit)

    # endregion

    # region Cycle Methods

    def _append_cycle(self) -> None:
        """Appends an idle cycle to the end of the circuit."""
        self._circuit.append([None] * self.num_qudits)

    def _insert_cycle(self, cycle: int) -> None:
        """Inserts an idle cycle in the circuit."""
        self._circuit.insert(cycle, [None] * self.num_qudits)

    def pop_cycle(self, cycle: int) -> None:
        """Pop a cycle from the circuit and all operations in it."""
        for qudit in self._circuit[cycle]:
            if qudit is not None:
                self.pop_gate(cycle, qudit)
        self._circuit.pop(cycle)

    def _is_cycle_idle(self, cycle: int) -> bool:
        """Return true if the cycle is idle, that is it contains no gates."""
        return all(q is None for q in self._circuit[cycle])

    def is_cycle_in_range(self, cycle: int) -> bool:
        """Return true if cycle is a valid in-range index in the circuit."""
        if not isinstance(cycle, int):
            raise TypeError("Expected int, got %s" % type(cycle))

        return cycle < self.cycles and cycle >= -self.cycles

    def is_cycle_unoccupied(self, cycle: int, location: Sequence[int]) -> bool:
        """
        Check if `cycle` is unoccupied for all qudits in `location`.

        Args:
            cycle (int): The cycle to check.

            location (Sequence[int]): The set of qudits to check.
        
        Raises:
            IndexError: If `cycle` is out of range.
        
        Examples:
            >>> circ = Circuit(2)
            >>> circ.append_gate(gates.H, [0])
            >>> circ.append_gate(gates.X, [0])
            >>> circ.append_gate(gates.Z, [1])
            >>> circ.is_cycle_unoccupied(0, [0])
            False
            >>> circ.is_cycle_unoccupied(1, [1])
            True
        """
        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')
        
        if not self.is_cycle_in_range(cycle):
            raise IndexError('Out-of-range cycle index: %d.' % cycle)

        for qudit_index in location:
            if self._circuit[cycle][qudit_index] is not None:
                return False

        return True

    def find_available_cycle(self, location: Sequence[int]) -> int:
        """
        Finds the first available cycle for qudits in `location`.
        
        An available cycle for `location` is one where it and all
        cycles after it are unoccupied for `location`.

        Args:
            localtion (Sequence[int]): Find a cycle for this location.
        
        Raises:
            ValueError: If no available cycle exists.

        Examples:
            >>> circ = Circuit(2)
            >>> circ.append_gate(gates.H, [0])
            >>> circ.find_available_cycle([1])
            0
            >>> circ.append_gate(gates.X, [0])
            >>> circ.append_gate(gates.Z, [1])
            >>> circ.find_available_cycle([1])
            1
        """

        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')
        
        if self.cycles == 0:
            raise ValueError("No available cycle.")

        # Iterate through cycles in reverse order
        for cycle in range(self.cycles - 1, -1, -1):
            # Find the first occupied cycle
            if not self.is_cycle_unoccupied(cycle, location):
                # The first available cycle is the previous one
                if cycle == self.cycles - 1:
                    raise ValueError("No available cycle.")
                return cycle + 1
        
        # If we didn't find an occupied cycle,
        # then they are all unoccupied.
        return 0
    
    def _find_available_or_append_cycle(self, location: Sequence[int]) -> int:
        """Find the first available cycle, if none exists append one."""
        try:
            return self.find_available_cycle(location)
        except ValueError:
            self._append_cycle()
            return self.cycles - 1

    # endregion

    # region Operation/Gate/Circuit Methods

    def is_point_in_range(self, point: tuple[int, int]) -> bool:
        """Return true if point is a valid in-range index in the circuit."""
        return ( self.is_cycle_in_range(point[0])
                 and self.is_qudit_in_range(point[1]) )

    def check_valid_operation(self, op: Operation) -> None:
        """Check that `op` can be applied to the circuit."""
        if not isinstance(op, Operation):
            raise TypeError('Expected Operation got %s.' % type(op))

        if not all([qudit < self.get_size() for qudit in op.location]):
            raise ValueError("Operation location mismatch with Circuit.")

        for op_radix, circ_radix_idx in zip(op.get_radixes(), op.location):
            if op_radix != self.radixes[circ_radix_idx]:
                raise ValueError('Operation radix mismatch with Circuit.')

    def get_operation(self, point: tuple[int, int]) -> Operation:
        """
        Retrieve the operation at the `point`.

        Args:
            point (tuple[int, int]): The circuit point position of the
                operation.
            
        Raises:
            IndexError: If no operation exists at the point specified.
        
        Returns:
            (Operation): The operation at `point`.
        
        Examples:
            >>> circuit = Circuit(2)
            >>> circuit.append_gate(H(), [0])
            >>> circuit.append_gate(CX(), [0, 1])
            >>> circuit.get_operation((1, 0))
            CX()@(0, 1)  # TODO: Update when __str__ figured out
        """
        if not self.is_point_in_range(point):
            raise IndexError("Out-of-range or invalid point.")

        op = self._circuit[point[0]][point[1]]

        if op is None:
            raise IndexError("No operation exists at the specified point.")
        
        return op
    
    def point(
        self,
        op: Operation | Gate,
        start: tuple[int, int] = (0, 0),
        end: Optional[tuple[int, int]] = None
    ) -> tuple[int, int]:
        """
        Return point of the first occurrence of `op`.

        Args:
            op (Operation | Gate): The operation or gate to find.

            start (tuple[int, int]): Circuit point to start searching
                from. Inclusive. (Default: Beginning of the circuit.)
            
            end (tuple[int, int] | None): Cycle index to stop searching at.
                Exclusive. (Default: End of the circuit.)
        
        Returns:
            The first point that contains `op`.
        
        Raises:
            ValueError: If `op` is not found.
        
            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.
        
        Examples:
            >>> circ = Circuit(1)
            >>> opH = Operation(H(), [0])
            >>> circ.append(opH)
            >>> circ.point(opH)
            (0, 0)
            >>> opX = Operation(X(), [0])
            >>> circ.point(opX)
            (1, 0)
        """
        if isinstance(op, Operation):
            self.check_valid_operation(op)
            if op.gate not in self._gate_set:
                raise ValueError("No such operation exists in the circuit.")
        elif isinstance(op, Gate):
            if op not in self._gate_set:
                raise ValueError("No such operation exists in the circuit.")
        else:
            raise TypeError("Expected gate or operation, got %s." % type(op))
        
        end = end or (self.get_num_cycles(), 0)

        if not self.is_point_in_range(start):
            raise IndexError("Out-of-range or invalid start point.")

        if not self.is_point_in_range(end):
            raise IndexError("Out-of-range or invalid end point.")

        if isinstance(op, Operation):
            qudit_index = op.location[0]
            for _p, _op in self.operations_on_qudit_with_points(qudit_index):
                if _op == op:
                    return _p

        raise ValueError("No such operation exists in the circuit.")

    def index(self, op: Operation | Gate, start: int = 0, end: int = -1) -> int:
        """
        Return index of `op`'s first occurrence in the circuit.

        Args:
            op (Operation | Gate): The operation or gate to find.

            start (int): Index to start searching from. Inclusive.
                (Default: 0)
            
            end (int): Index to stop searching at. Inclusive.
                (Default: -1)
        
        Returns:
            The first index that contains `op`.
        
        Raises:
            ValueError: If `op` is not found.
        
            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.
        
        Examples:
            >>> circ = Circuit(1)
            >>> opH = Operation(H(), [0])
            >>> circ.append(opH)
            >>> circ.index(opH)
            0
            >>> opX = Operation(X(), [0])
            >>> circ.index(opX)
            1
        """
        if isinstance(op, Operation):
            self.check_valid_operation(op)
        elif not isinstance(op, Gate):
            raise TypeError("Expected gate or operation, got %s." % type(op))

        end = end + 1 if end >= 0 else end - 1
        
        for i, _op in enumerate(self[start:end]):
            if isinstance(op, Operation) and op == _op:
                return i + start
            if isinstance(op, Gate) and op == _op.gate:
                return i + start

        raise ValueError("No such operation exists in the circuit.")

    def append(self, op: Operation) -> None:
        """
        Append `op` to the end of the circuit.

        Args:
            op (Operation): The operation to append.
        
        Raises:
            ValueError: If `op` cannot be placed on the circuit due to
                either an invalid location or gate radix mismatch.
        
        Notes:
            Due to the circuit being represented as a matrix,
            `circuit.append(op)` does not imply `circuit[-1] == op`,
            but it implies op is in the last cycle of circuit:
            `circuit.cycle(op, -1) == -1`
        
        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op) # Appends a Hadamard gate to qudit 0.
        """
        self.check_valid_operation(op)

        if op.gate not in self._gate_set:
            self._gate_set[op.gate] = 0
        self._gate_set[op.gate] += 1

        cycle = self._find_available_or_append_cycle(op.location)

        for qudit_index in op.location:
            self._circuit[cycle][qudit_index] = op
 
    def append_gate(
        self,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        """
        Append the gate object to the circuit on the qudits described by
        location. Optionally, you can specify parameters for the gate. By
        default, the params are zeroed.

        Args:
            gate (Gate): The gate to append.

            location (Sequence[int]): Apply the gate to this set of qudits.

            params (Sequence[float]): The gate's parameters.

        Examples:
            >>> circ = Circuit(1)
            >>> # Append a Hadamard gate to qudit 0.
            >>> circ.append_gate(gates.H, [0])
        """

        if not isinstance(gate, Gate):
            raise TypeError('Invalid gate.')

        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        if len(location) != gate.get_size():
            raise ValueError('Gate and location size mismatch.')

        for gate_radix, circ_radix_idx in zip(gate.get_radixes(), location):
            if gate_radix != self.qudit_radixes[circ_radix_idx]:
                raise ValueError('Gate and location radix mismatch.')

        if params is None:
            params = [0.0] * gate.get_num_params()

        if len(params) != gate.get_num_params():
            raise ValueError('Gate and parameter mismatch.')

        if gate not in self._gate_set:
            self._gate_set[gate] = 0
        self._gate_set[gate] += 1

        cell = Operation(gate, location, params)

        cycle = self.find_available_cycle(location)

        if cycle == -1:
            self.append_cycle()

        for qudit_index in location:
            self._circuit[cycle][qudit_index] = cell

    def append_circuit(
        self,
        circuit: Circuit,
        location: Sequence[int]
    ) -> None:
        pass
    
    def extend(self, ops: Iterable[Operation]) -> None:
        """
        Append all operations in `ops` to the circuit.

        Args:
            ops (Operation): The operations to append.
        
        Examples:
            >>> circ = Circuit(1)
            >>> opH = Operation(H(), [0])
            >>> opX = Operation(X(), [0])
            >>> circ.extend([opH, opX])
            >>> circ.index(opH)
            0
            >>> circ.index(opX)
            1
        
        Notes:
            See `append` for more info.
        """
        for op in ops:
            self.append(op)
    
    def insert(self, cycle: int, op: Operation) -> None:
        """
        Insert `op` in the circuit at the specified cycle.

        After this if cycle was in range, you can expect:
        all([self.get_gate(cycle, idx) == gate for idx in location])

        Args:
            cycle (int): The cycle to insert the operation.

            op (Operation): The operation to insert.

        Examples:
            >>> circ = Circuit(1)
            >>> opX = Operation(X(), [0])
            >>> opH = Operation(H(), [0])
            >>> circ.append(opX)
            >>> circ.insert(opH, 0)
            >>> circ.cycle(opH)
            0
        
        Notes:
            Clamps cycle to be in range.
        """
        self.check_valid_operation(op)

        if self.is_cycle_in_range(cycle):
            if cycle < 0:
                cycle = 0
            else:
                self.append(op)
                return

        if op.gate not in self._gate_set:
            self._gate_set[op.gate] = 0
        self._gate_set[op.gate] += 1

        cycle = self.cycles + cycle if cycle < 0 else cycle

        if not self.is_cycle_unoccupied(cycle, op.location):
            self._insert_cycle(cycle)

        for qudit_index in op.location:
            self._circuit[cycle][qudit_index] = op
   
    def insert_gate(
        self,
        cycle: int,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        """
        Insert the gate object in the circuit on the qudits described by
        location at the cycle specified. Optionally, you can specify
        parameters for the gate. By default, the params are zeroed. After this,
        you can expect: all( [ self.get_gate(cycle, idx) == gate for idx in
        location ] )

        Args:
            gate (Gate): The gate to insert.

            cycle (int): The cycle to insert the gate.

            location (Sequence[int]): Apply the gate to this set of qudits.

            params (Sequence[float]): The gate's parameters.

        Raises:
            IndexError: If the specified cycle doesn't exist.

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

        if cycle < -self.cycles or cycle >= self.cycles:
            raise IndexError('Invalid cycle.')

        if not is_valid_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        if len(location) != gate.get_size():
            raise ValueError('Gate and location size mismatch.')

        for gate_radix, circ_radix_idx in zip(gate.get_radixes(), location):
            if gate_radix != self.qudit_radixes[circ_radix_idx]:
                raise ValueError('Gate and location radix mismatch.')

        if params is None:
            params = [0.0] * gate.get_num_params()

        if len(params) != gate.get_num_params():
            raise ValueError('Gate and parameter mismatch.')

        if cycle < 0:
            cycle = self.cycles - cycle

        if gate not in self._gate_set:
            self._gate_set[gate] = 0
        self._gate_set[gate] += 1

        cell = Operation(gate, location, params)

        if not self.is_cycle_available(cycle, location):
            self.insert_cycle(cycle)

        for qudit_index in location:
            self._circuit[cycle][qudit_index] = cell

    def insert_circuit(
        self,
        cycle: int,
        circuit: Circuit,
        location: Sequence[int]
    ) -> None:
        pass

    def remove(self, op: Operation | Gate) -> None:
        """
        Removes the first occurrence of `op` in the circuit.

        Args:
            op (Operation): The Operation to remove.

        Raises:
            ValueError: If the `op` doesn't exist in the circuit.
        
            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.

        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op)
            >>> circ.num_gates
            1
            >>> circ.remove(op)
            >>> circ.num_gates
            0
        """
        self.check_valid_operation(op)
        cycle = self.cycle(op)
        self.pop((cycle, op.location[0]))
    
    def count(self, op: Operation | Gate) -> int:
        """
        Count the number of times `op` in the circuit.

        Args:
            op (Operation): The Operation to count.

        Raises:
            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.

        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op)
            >>> circ.count(op)
            1
            >>> circ.append(op)
            >>> circ.count(op)
            2
        """
        self.check_valid_operation(op)

        if op.gate not in self._gate_set:
            return 0
        
        count = 0
        qudit_index = op.location[0]
        for cycle in range(self.cycles):
            _op = self._circuit[cycle][qudit_index]
            if _op is not None and _op == op:
                count += 1
        return count
    
    def pop(self, point: Optional[tuple[int, int]] = None) -> Operation:
        """
        Pop the operation at `point`, defaults to last operation.

        Args:
            point (Optional[tuple[int, int]]): The cycle and qudit index
                to pop from.
            
        Raises:
            IndexError: If the `point` is out-of-range, or if no operation
                exists at `point`.

        Examples:
            >>> circ = Circuit(1)
            >>> circ.append_gate(H(), [0])
            >>> circ.get_num_gates()
            1
            >>> circ.pop(0, 0)
            >>> circ.get_num_gates()
            0
        """
        cycle = None
        qudit = None

        # Use given point
        if point is not None:
            if not self.is_point_in_range(point):
                raise IndexError('Out-of-range point: %s.' % str(point))
            cycle = point[0]
            qudit = point[1]
        
        # Or find last gate in simulation order
        else:
            cycle = self.get_num_cycles() - 1
            for i, op in enumerate(reversed(self._circuit[cycle])):
                if op is not None:
                    qudit = self.get_size() - 1 - i
        
        if cycle is None or qudit is None:
            raise IndexError("Out-of-range point: %s." % str(point))
        
        op = self._circuit[cycle][qudit]

        if op is None:
            raise IndexError("No operation exists at point: %s." % str(point))
            
        for qudit_index in op.location:
            self._circuit[cycle][qudit] = None
        
        self._gate_set[op.gate] -= 1
        if self._gate_set[op.gate] <= 0:
            del self._gate_set[op.gate]
        
        if self._is_cycle_idle(cycle):
            self.pop_cycle(cycle)
    
    def replace(self, cycle: int, op: Operation) -> None:
        pass

    def replace_gate(
        self,
        cycle: int,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        pass

    def replace_with_circuit(
        self,
        cycle: int,
        cirucit: Circuit,
        location: Sequence[int],
    ) -> None:
        pass

    def copy(self) -> Circuit:
        pass

    def slice(self, points: Sequence[tuple[int, int]]) -> Circuit:
        pass
    
    def clear(self) -> None:
        pass

    def operations(self) -> Iterator[Operation]:
        pass

    def operations_with_points(self) -> Iterator[tuple[tuple[int, int], Operation]]:
        pass

    def operations_on_qudit(self, qudit_index: int) -> Iterator[Operation]:
        pass

    def operations_on_qudit_with_points(self) -> Iterator[tuple[tuple[int, int], Operation]]:
        pass

    def gates(self) -> Iterator[Gate]:
        pass

    # endregion

    # region Parameter Methods

    def get_param(self, param_index: int) -> float:
        """Return the parameter at param_index."""
        cycle, qudit, param = self.get_param_location(param_index)
        return self[cycle, qudit].params[param]
        
    def set_param(self, param_index: int, value: float) -> None:
        """Set a circuit parameter"""
        cycle, qudit, param = self.get_param_location(param_index)
        self[cycle, qudit].params[param] = value

    def freeze_param(self, param_index: int) -> None:
        """Freeze a circuit parameter to its current value."""
        cycle, qudit, param = self.get_param_location(param_index)
        op = self[cycle, qudit]
        gate = op.gate.with_frozen_params({param: op.params[param]})
        params = op.params.copy()
        params.pop(param)    
        self.replace_gate(cycle, gate, op.location, params)

    def get_param_location(self, param_index: int) -> tuple[int, int, int]:
        """
        Converts a param_index to a cycle, qudit, and operation-param index.

        Args:
            param_index (int): The parameter index to convert.

        Returns:
            (tuple[int, int, int]): A tuple of cycle_index, qudit_index,
                and operation-param index. The operation the parameter
                belongs to will be at circuit[cycle_index, qudit_index].
                This parameter in that operation is indexed by
                the operation-param index.

        Examples:
            >>> circ = Circuit(1)
            >>> circ.append_gate(U3(), [0])
            >>> circ.append_gate(U3(), [0])
            >>> circ.get_num_params()
            6
            >>> circ.get_param_location(4)
            (1, 0, 1)
        """
        count = 0
        for cycle, op in self.operations_with_cycles():
            count += len(op.params)
            if count > param_index:
                param = param_index - (count - len(op.params))
                return (cycle, op.location[0], param)
    
    # endregion

    # region Circuit Logic Methods
    
    def get_inverse(self) -> Circuit:
        pass
    
    def get_dagger(self) -> Circuit:
        return self.get_inverse()

    def renumber_qudits(self, qudit_permutation: Sequence[int]) -> None:
        """
        Permute the qudits in the circuit.

        Args:
            qudit_index_map (dict[int, int]): A map from qudit indices
                to qudit indices.
        
        Raises:
            IndexError: If any of the indices are out of range.
        """
        if not is_sequence(qudit_permutation):
            raise TypeError(
                "Expected sequence of integers"
                ", got %s" % type(qudit_permutation)
            )
        
        if len(qudit_permutation) != self.get_size():
            raise ValueError(
                "Expected qudit_permutation length equal to circuit size:"
                "%d, got %d" % (self.get_size(), len(qudit_permutation))
            )
        
        if len(qudit_permutation) != len(set(qudit_permutation)):
            raise ValueError("Invalid permutation.")

        qudit_permutation = [int(q) for q in qudit_permutation]

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """
        Return the unitary matrix of the circuit.

        Args:
            params (Sequence[float]): Optionally specify parameters
                overriding the ones stored in the circuit. (Default:
                use parameters already in circuit.)
        
        Returns:
            The UnitaryMatrix object that the circuit implements.
        
        Raises:
            ValueError: If parameters are specified and invalid.
        
        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op)
            >>> circ.get_unitary() == H().get_unitary()
            True
        """
        if params:
            self.check_parameters(params)
            param_index = 0

        utry = UnitaryBuilder(self.get_size(), self.get_radixes())
        
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                utry.apply_right(op.get_unitary(gparams))
                param_index += op.get_num_params()
            else:
                utry.apply_right(op.get_unitary())

        return utry.get_unitary()
    
    def get_statevector(self, in_state: StateVector) -> StateVector:
        pass

    def optimize(self, method: str, **kwargs: Any) -> None:
        pass
    
    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Return the gradient of the circuit."""
        if params:
            self.check_parameters(params)
            param_index = 0

        left = UnitaryBuilder(self.get_size(), self.get_radixes())
        right = UnitaryBuilder(self.get_size(), self.get_radixes())
        
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_right(op.get_unitary(gparams))
                param_index += op.get_num_params()
            else:
                right.apply_right(op.get_unitary())

        grads = []
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_left(op.get_unitary(gparams), inverse = True)
                param_index += op.get_num_params()
                grad = op.get_grad(gparams)
                grads.append( left.get_unitary() @ grad @ right.get_unitary() )
                left.apply_right(op.get_unitary(gparams))
            else:
                right.apply_left(op.get_unitary(), inverse = True)
                grad = op.get_grad()
                grads.append( left.get_unitary() @ grad @ right.get_unitary() )
                left.apply_right(op.get_unitary(gparams))
        
        return np.array( grads )

    def get_unitary_and_grad(self, params: Sequence[float] = []) -> tuple[UnitaryMatrix, np.ndarray]:
        """Return the unitary and gradient of the circuit."""
        if params:
            self.check_parameters(params)
            param_index = 0

        left = UnitaryBuilder(self.get_size(), self.get_radixes())
        right = UnitaryBuilder(self.get_size(), self.get_radixes())
        
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_right(op.get_unitary(gparams))
                param_index += op.get_num_params()
            else:
                right.apply_right(op.get_unitary())

        grads = []
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_left(op.get_unitary(gparams), inverse = True)
                param_index += op.get_num_params()
                grad = op.get_grad(gparams)
                grads.append( left.get_unitary() @ grad @ right.get_unitary() )
                left.apply_right(op.get_unitary(gparams))
            else:
                right.apply_left(op.get_unitary(), inverse = True)
                grad = op.get_grad()
                grads.append( left.get_unitary() @ grad @ right.get_unitary() )
                left.apply_right(op.get_unitary(gparams))
        
        return left.get_unitary(), np.array( grads )
    
    # endregion

    # region Operation Iteration and Container Methods

    def __getitem__(
        self,
        points: tuple[int, int] | Sequence[tuple[int, int]] | slice
    ) -> Operation | Circuit:
        """
        Retrieve an operation from a point or a circuit from a sequence of points.

        Args:
            points (tuple[int, int] | Sequence[tuple[int, int]] | slice):
                Either a point, a list of points, or a slice of points.
                If a point is given, returns the operation at that point.
                If a list of points is given, return a circuit containing
                the operations at those points. If a slice is given,
                return a circuit with the operations that are within the
                slice.
        
        Returns:
            (Operation | Circuit): Either a specific operation is returned
                or a Circuit if multiple points are specified.
        
        Raises:
            IndexError: If a specified point does not contain an operation.
        
        Notes:
            If a circuit is returned, it is not a view but rather a copy.
        """
        
        if isinstance(points, tuple):
            if all(isinstance(q, int) for q in points) and len(points) == 2:
                return self.get_operation(points)

        elif isinstance(points, slice) or is_sequence(points):
            return self.slice(points)

        else:
            raise TypeError(
                "Invalid index type. Expected point"
                ", sequence of points, or slice"
                ", got %s" % type(points)
            )

    class CircuitIterator(Iterator[Operation]):
        def __init__(self, circuit: list[list[Operation | None]], reversed: bool = False) -> None:
            self.circuit = circuit
            self.reversed = reversed
            self.max_cycle = len(circuit)
            self.max_qudit = 0 if self.max_cycle == 0 else len(circuit[0])
            self.cycle = 0 if not reversed else self.max_cycle - 1
            self.qudit = 0 if not reversed else self.max_qudit - 1
            self.qudits_to_skip: set[int] = set()

        def increment_iter(self) -> None:
            self.qudit += 1
            while self.qudit in self.qudits_to_skip:
                self.qudit += 1
            if self.qudit >= self.max_qudit:
                self.qudit = 0
                self.cycle += 1
                self.qudits_to_skip = set()
            if self.cycle >= self.max_cycle:
                raise StopIteration
        
        def decrement_iter(self) -> None:
            self.qudit -= 1
            while self.qudit in self.qudits_to_skip:
                self.qudit -= 1
            if self.qudit < 0:
                self.qudit = self.max_qudit - 1
                self.cycle -= 1
                self.qudits_to_skip = set()
            if self.cycle < 0:
                raise StopIteration

        def dereference(self) -> Operation | None:
            return self.circuit[self.cycle][self.qudit]

        def __iter__(self) -> Iterator[Operation]:
            return self

        def __next__(self) -> Operation:
            while self.dereference() is None:
                if not self.reversed:
                    self.increment_iter()
                else:
                    self.decrement_iter()
            op = self.dereference()  # type: ignore
            self.qudits_to_skip.update(op.location)
            return op

    class QuditIterator(Iterator[Operation]):
        def __init__(self, qudit: int, circuit: list[list[Operation | None]], reversed: bool = False) -> None:
            self.circuit = circuit
            self.reversed = reversed
            self.qudit = qudit
            self.max_cycle = len(circuit)
            self.cycle = 0 if not reversed else self.max_cycle - 1

            self.max_qudit = 0 if self.max_cycle == 0 else len(circuit[0])
            if qudit >= self.max_qudit or qudit < 0:
                raise IndexError("Invalid qudit index for iterator.")

        def increment_iter(self) -> None:
            self.cycle += 1
            if self.cycle >= self.max_cycle:
                raise StopIteration
        
        def decrement_iter(self) -> None:
            self.cycle -= 1
            if self.cycle < 0:
                raise StopIteration

        def dereference(self) -> Operation | None:
            return self.circuit[self.cycle][self.qudit]

        def __iter__(self) -> Iterator[Operation]:
            return self

        def __next__(self) -> Operation:
            while self.dereference() is None:
                if not self.reversed:
                    self.increment_iter()
                else:
                    self.decrement_iter()
            op = self.dereference()  # type: ignore
            return op

    def __iter__(self) -> Iterator[Operation]:
        return self.CircuitIterator(self._circuit)

    def __reversed__(self) -> Iterator[Operation]:
        return self.CircuitIterator(self._circuit, reversed = True)

    def __contains__(self, op: Operation | Gate) -> bool:
        """Return true if `op` is in the circuit."""
        if isinstance(op, Operation):
            try:
                self.check_valid_operation(op)
            except ValueError:
                return False

            if op.gate not in self._gate_set:
                return False
            
            for _op in self.operations_on_qudit(op.location[0]):
                if op == _op:
                    return True
            
            return False

        elif isinstance(op, Gate):
            return op in self._gate_set

        else:
            raise TypeError("Expected gate or operation, got %s." % type(op))

    def __len__(self) -> int:
        """Return the number of operations in the circuit."""
        return self.get_num_operations()
    
    # endregion

    # region Operator Overloads

    def __invert__(self) -> Circuit:
        """Invert the circuit."""
        return self.get_inverse()

    def __eq__(self, rhs: Circuit) -> bool:
        """
        Check for circuit equality.
        
        Two circuits are equal if:
            1) They have the same number of operations.
            2) All qudit radixes are equal.
            2) all operations in simulation order are equal.
        """
        if not isinstance(rhs, Circuit):
            raise NotImplemented

        if self is rhs:
            return True
        
        if self._gate_set != rhs._gate_set:
            return False
        
        for r1, r2 in zip(self.get_radixes(), rhs.get_radixes()):
            if r1 != r2:
                return False
        
        return all(op1 == op2 for op1, op2 in zip(self, rhs))

    def __ne__(self, rhs: Circuit) -> bool:
        """Check for circuit inequality, see __eq__ for more info."""
        return not self == rhs

    def __add__(self, rhs: Circuit) -> Circuit:
        pass

    def __mul__(self, rhs: int) -> Circuit:
        pass
    
    def __radd__(self, lhs: Circuit) -> Circuit:
        pass
    
    def __rmul__(self, lhs: int) -> Circuit:
        pass
    
    def __iadd__(self, rhs: Circuit) -> None:
        pass
    
    def __imul__(self, rhs: int) -> None:
        pass
    
    # endregion

    # region IO Methods

    def __str__(self) -> str:
        pass
    
    def __repr__(self) -> str:
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
    
    # endregion

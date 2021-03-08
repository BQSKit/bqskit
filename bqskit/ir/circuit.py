"""This module implements the Circuit class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import overload
from typing import Sequence
from typing import Union

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.statemap import StateVectorMap
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_valid_location
from bqskit.utils.typing import is_valid_radixes


_logger = logging.getLogger(__name__)


class Circuit(DifferentiableUnitary, StateVectorMap, Collection[Operation]):
    """
    Circuit class.

    A Circuit is a quantum program composed of operation objects.

    The operations are organized in 2-dimensions, and are indexed by
    a CircuitPoint.

    Invariants:
        1. A circuit method will never complete with an idle cycle.
            An idle cycle is one that contains no gates.

        2. No one logical operation will ever be pointed to from more than
            one cycle. The Operation object is a CachedClass, so it may appear
            multiple times throughout the circuit, but never as one logical
            operation across multiple cycles.
    """

    def __init__(self, size: int, radixes: Sequence[int] = []) -> None:
        """
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
                ', got %d.' % size,
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
        for _, count in self._gate_set.items():
            num_gates_acm += count
        return num_gates_acm

    def get_num_cycles(self) -> int:
        """Return the number of cycles in the circuit."""
        return len(self._circuit)

    def get_params(self) -> list[float]:
        """Returns the stored parameters for the circuit."""
        return sum([op.params for op in self])  # type: ignore

    def get_depth(self) -> int:
        """Return the length of the critical path in the circuit."""
        qudit_depths = np.zeros(self.get_size())
        for op in self:
            new_depth = max(qudit_depths[op.location]) + 1
            qudit_depths[op.location] = new_depth
        return max(qudit_depths)

    def get_parallelism(self) -> float:
        """Calculate the amount of parallelism in the circuit."""
        return self.get_num_operations() / self.get_depth()

    def get_coupling_graph(self) -> set[tuple[int, int]]:
        """
        The qudit connectivity in the circuit.

        Returns:
            (set[tuple[int, int]]): The coupling graph required by
                the circuit. The graph is returned as a edge list.

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

    def append_qudit(self, radix: int = 2) -> None:
        """
        Append a qudit to the circuit.

        Args:
            radix (int): The radix of the qudit. (Default: qubit)

        Raises:
            ValueError: If `radix` is < 2
        """

        if radix < 2:
            raise ValueError('Expected radix to be > 2, got %d' % radix)

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

    def insert_qudit(self, qudit_index: int, radix: int = 2) -> None:
        """
        Insert a qudit in to the circuit.

        Args:
            qudit_index (int): The index where to insert the qudit.

            radix (int): The radix of the qudit. (Default: qubit)

        Raises:
            ValueError: If `radix` is < 2.
        """

        if radix < 2:
            raise ValueError('Expected radix to be > 2, got %d' % radix)

        self.size += 1
        self.radixes.insert(qudit_index, radix)

        for cycle in self._circuit:
            cycle.insert(qudit_index, None)

    def pop_qudit(self, qudit_index: int) -> None:
        """Pop a qudit from the circuit and all gates attached to it."""
        for cycle_index, cycle in enumerate(self._circuit):
            if cycle[qudit_index] is not None:
                self.pop((cycle_index, qudit_index))
            cycle.pop(qudit_index)

    def is_qudit_in_range(self, qudit_index: int) -> bool:
        """Return true if qudit index is in-range for the circuit."""
        return (
            qudit_index < self.get_size()
            and qudit_index >= -self.get_size()
        )

    def is_qudit_idle(self, qudit_index: int) -> bool:
        """Return true if the qudit is idle."""
        return all(cycle[qudit_index] is None for cycle in self._circuit)

    # endregion

    # region Cycle Methods

    def _append_cycle(self) -> None:
        """Appends an idle cycle to the end of the circuit."""
        self._circuit.append([None] * self.get_size())

    def _insert_cycle(self, cycle_index: int) -> None:
        """Inserts an idle cycle in the circuit."""
        self._circuit.insert(cycle_index, [None] * self.get_size())

    def pop_cycle(self, cycle_index: int) -> None:
        """Pop a cycle from the circuit and all operations in it."""
        for op in self._circuit[cycle_index]:
            if op is not None:
                self._gate_set[op.gate] -= 1
                if self._gate_set[op.gate] <= 0:
                    del self._gate_set[op.gate]
        self._circuit.pop(cycle_index)

    def _is_cycle_idle(self, cycle_index: int) -> bool:
        """Return true if the cycle is idle, that is it contains no gates."""
        return all(op is None for op in self._circuit[cycle_index])

    def is_cycle_in_range(self, cycle_index: int) -> bool:
        """Return true if cycle is a valid in-range index in the circuit."""
        return (
            cycle_index < self.get_num_cycles()
            and cycle_index >= -self.get_num_cycles()
        )

    def is_cycle_unoccupied(
        self, cycle_index: int,
        location: Sequence[int],
    ) -> bool:
        """
        Check if a cycle is unoccupied for all qudits in `location`.

        Args:
            cycle_index (int): The cycle to check.

            location (Sequence[int]): The set of qudits to check.

        Raises:
            IndexError: If `cycle_index` is out of range.

        Examples:
            >>> circ = Circuit(2)
            >>> circ.append_gate(H(), [0])
            >>> circ.append_gate(X(), [0])
            >>> circ.append_gate(Z(), [1])
            >>> circ.is_cycle_unoccupied(0, [0])
            False
            >>> circ.is_cycle_unoccupied(1, [1])
            True
        """
        if not is_valid_location(location, self.get_size()):
            raise TypeError('Invalid location.')

        if not self.is_cycle_in_range(cycle_index):
            raise IndexError('Out-of-range cycle index: %d.' % cycle_index)

        for qudit_index in location:
            if self._circuit[cycle_index][qudit_index] is not None:
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
            >>> circ.append_gate(H(), [0])
            >>> circ.find_available_cycle([1])
            0
            >>> circ.append_gate(X(), [0])
            >>> circ.append_gate(Z(), [1])
            >>> circ.find_available_cycle([1])
            1
        """

        if not is_valid_location(location, self.get_size()):
            raise TypeError('Invalid location.')

        if self.get_num_cycles() == 0:
            raise ValueError('No available cycle.')

        # Iterate through cycles in reverse order
        for cycle in range(self.get_num_cycles() - 1, -1, -1):
            # Find the first occupied cycle
            if not self.is_cycle_unoccupied(cycle, location):
                # The first available cycle is the previous one
                if cycle == self.get_num_cycles() - 1:
                    raise ValueError('No available cycle.')
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
            return self.get_num_cycles() - 1

    # endregion

    # region Operation/Gate/Circuit Methods

    def is_point_in_range(self, point: CircuitPointLike) -> bool:
        """Return true if point is a valid in-range index in the circuit."""
        return (
            self.is_cycle_in_range(point[0])
            and self.is_qudit_in_range(point[1])
        )

    def check_valid_operation(self, op: Operation) -> None:
        """Check that `op` can be applied to the circuit."""
        if not isinstance(op, Operation):
            raise TypeError('Expected Operation got %s.' % type(op))

        if not all([qudit < self.get_size() for qudit in op.location]):
            raise ValueError('Operation location mismatch with Circuit.')

        for op_radix, circ_radix_idx in zip(op.get_radixes(), op.location):
            if op_radix != self.get_radixes()[circ_radix_idx]:
                raise ValueError('Operation radix mismatch with Circuit.')

    def get_operation(self, point: CircuitPointLike) -> Operation:
        """
        Retrieve the operation at the `point`.

        Args:
            point (CircuitPointLike): The circuit point position of the
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
            raise IndexError('Out-of-range or invalid point.')

        op = self._circuit[point[0]][point[1]]

        if op is None:
            raise IndexError('No operation exists at the specified point.')

        return op

    def point(
        self,
        op: Operation | Gate,
        start: CircuitPointLike = (0, 0),
        end: CircuitPointLike | None = None,
    ) -> CircuitPoint:
        """
        Return point of the first occurrence of `op`.

        Args:
            op (Operation | Gate): The operation or gate to find.

            start (CircuitPointLike): Circuit point to start searching
                from. Inclusive. (Default: Beginning of the circuit.)

            end (CircuitPointLike | None): Cycle index to stop searching at.
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
            CircuitPoint(cycle=0, qudit=0)
            >>> opX = Operation(X(), [0])
            >>> circ.point(opX)
            CircuitPoint(cycle=1, qudit=0)
        """
        if isinstance(op, Operation):
            self.check_valid_operation(op)
            if op.gate not in self._gate_set:
                raise ValueError('No such operation exists in the circuit.')
        elif isinstance(op, Gate):
            if op not in self._gate_set:
                raise ValueError('No such operation exists in the circuit.')
        else:
            raise TypeError('Expected gate or operation, got %s.' % type(op))

        end = end or (self.get_num_cycles(), 0)

        if not self.is_point_in_range(start):
            raise IndexError('Out-of-range or invalid start point.')

        if not self.is_point_in_range(end):
            raise IndexError('Out-of-range or invalid end point.')

        if isinstance(op, Operation):
            qudit_index = op.location[0]
            for i, cycle in enumerate(self._circuit[start[0]:end[0]]):
                if cycle[qudit_index] is not None and cycle[qudit_index] == op:
                    return CircuitPoint(start[0] + i, qudit_index)
        else:
            for i, cycle in enumerate(self._circuit[start[0]:end[0]]):
                for q, _op in enumerate(cycle[start[1]:end[1]]):
                    if _op is not None and _op.gate == op:
                        return CircuitPoint(start[0] + i, start[1] + q)

        raise ValueError('No such operation exists in the circuit.')

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
            `circuit.point(op, -1) == -1`

        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op) # Appends a Hadamard gate to qudit 0.
        """
        self.check_valid_operation(op)

        if op.gate not in self._gate_set:
            self._gate_set[op.gate] = 0
        self._gate_set[op.gate] += 1

        cycle_index = self._find_available_or_append_cycle(op.location)

        for qudit_index in op.location:
            self._circuit[cycle_index][qudit_index] = op

    def append_gate(
        self,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        """
        Append the gate object to the circuit on the qudits in location.

        Args:
            gate (Gate): The gate to append.

            location (Sequence[int]): Apply the gate to this set of qudits.

            params (Sequence[float]): The gate's parameters.
                (Default: all zeros)

        Examples:
            >>> circ = Circuit(1)
            >>> # Append a Hadamard gate to qudit 0.
            >>> circ.append_gate(H(), [0])
        """
        params = params or [0.0] * gate.get_num_params()
        self.append(Operation(gate, location, params))

    def append_circuit(
        self,
        circuit: Circuit,
        location: Sequence[int],
    ) -> None:
        """Append `circuit` at the qudit location specified."""
        if circuit.get_size() != len(location):
            raise ValueError('Circuit and location size mismatch.')

        for op in circuit:
            mapped_location = [location[q] for q in op.location]
            self.append(Operation(op.gate, mapped_location, op.params))

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

    def insert(self, cycle_index: int, op: Operation) -> None:
        """
        Insert `op` in the circuit at the specified cycle.

        After this, if cycle was in range, you can expect:
        `all([self[cycle_index, idx] == op for idx in op.location])`

        Args:
            cycle_index (int): The cycle to insert the operation.

            op (Operation): The operation to insert.

        Raises:
            ValueError: If `op` cannot be placed on the circuit due to
                either an invalid location or gate radix mismatch.

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

        if self.is_cycle_in_range(cycle_index):
            if cycle_index < -self.get_num_cycles():
                cycle_index = 0
            else:
                self.append(op)
                return

        if op.gate not in self._gate_set:
            self._gate_set[op.gate] = 0
        self._gate_set[op.gate] += 1

        if not self.is_cycle_unoccupied(cycle_index, op.location):
            self._insert_cycle(cycle_index)

        for qudit_index in op.location:
            self._circuit[cycle_index][qudit_index] = op

    def insert_gate(
        self,
        cycle_index: int,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        """
        Insert the gate object in the circuit on the qudits in location.

        After this, if cycle was in range, you can expect:
        `all([self[cycle_index, idx].gate == gate for idx in location])`

        Args:
            cycle_index (int): The cycle to insert the gate.

            gate (Gate): The gate to insert.

            location (Sequence[int]): Apply the gate to this set of qudits.

            params (Sequence[float]): The gate's parameters.

        Raises:
            IndexError: If the specified cycle doesn't exist.

            ValueError: If `gate` cannot be placed on the circuit due to
                either an invalid location or gate radix mismatch.
        """

        params = params or [0.0] * gate.get_num_params()
        self.insert(cycle_index, Operation(gate, location, params))

    def insert_circuit(
        self,
        cycle_index: int,
        circuit: Circuit,
        location: Sequence[int],
    ) -> None:
        """Insert `circuit` at the qudit location specified."""
        if circuit.get_size() != len(location):
            raise ValueError('Circuit and location size mismatch.')

        for op in reversed(circuit):
            mapped_location = [location[q] for q in op.location]
            self.insert(
                cycle_index,
                Operation(
                    op.gate,
                    mapped_location,
                    op.params,
                ),
            )

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
        self.pop(self.point(op))

    def count(self, op: Operation | Gate) -> int:
        """
        Count the number of times `op` occurs in the circuit.

        Args:
            op (Operation): The operation or gate to count.

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

        count = 0
        if isinstance(op, Operation):
            self.check_valid_operation(op)
            if op.gate not in self._gate_set:
                return 0

            qudit_index = op.location[0]
            for _op in self.operations_on_qudit(qudit_index):
                if _op == op:
                    count += 1
        elif isinstance(op, Gate):
            if op not in self._gate_set:
                return 0
            return self._gate_set[op]
        else:
            raise TypeError('Expected gate or operation, got %s.' % type(op))

        return count

    def pop(self, point: CircuitPointLike | None = None) -> Operation:
        """
        Pop the operation at `point`, defaults to last operation.

        Args:
            point (CircuitPoint | None): The cycle and qudit index
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

        # Use given point
        if point is not None:
            if not self.is_point_in_range(point):
                raise IndexError('Out-of-range point: %s.' % str(point))

        # Or find last gate in simulation order
        else:
            cycle_index = self.get_num_cycles() - 1
            for i, op in enumerate(reversed(self._circuit[cycle_index])):
                if op is not None:
                    point = (cycle_index, self.get_size() - 1 - i)
                    break

        if point is None:
            raise IndexError('Pop from empty circuit.')

        op = self[point]

        for qudit_index in op.location:
            self._circuit[point[0]][qudit_index] = None

        self._gate_set[op.gate] -= 1
        if self._gate_set[op.gate] <= 0:
            del self._gate_set[op.gate]

        if self._is_cycle_idle(point[0]):
            self.pop_cycle(point[0])

        return op

    def batch_pop(self, points: Sequence[CircuitPointLike]) -> Circuit:
        """
        Pop all operatons at `points` at once.

        Args:
            points (Sequence[CircuitPointLike]): Remove operations
                at these points all at the same time.

        Returns:
            (Circuit): The circuit formed from all the popped operations.

        Raises:
            IndexError: If any of `points` are invalid or out-of-range.
        """
        for point in points:
            if not self.is_point_in_range(point):
                raise IndexError('Out-of-range point.')
            if self._circuit[point[0]][point[1]] is None:
                raise IndexError('Invalid point.')

        # Sort points and collect operations and qudits
        points = sorted(points)
        ops = list({self[point] for point in points})
        qudits = sorted(list({point[1] for point in points}))

        # Pop gates, tracking if the circuit popped a cycle
        num_cycles = self.get_num_cycles()
        for i, point in enumerate(points):

            try:
                self.pop(point)
            except IndexError:  # Silently discard multi-points-one-op errors
                pass

            if num_cycles != self.get_num_cycles():
                num_cycles = self.get_num_cycles()
                points[i + 1:] = [(point[0] - 1, point[1])
                                  for point in points[i + 1:]]

        # Form new circuit and return
        radixes = [self.get_radixes()[q] for q in qudits]
        circuit = Circuit(len(radixes), radixes)
        for op in ops:
            location = [qudits.index(q) for q in op.location]
            circuit.append(Operation(op.gate, location, op.params))
        return circuit

    def replace(self, point: CircuitPointLike, op: Operation) -> None:
        """
        Replace the operation at `point` with `op`.

        Args:
            point (CircuitPointLike): The index of the operation to
                replace.

            op (Operation): The to-be-inserted operation.

        Raises:
            IndexError: If there is no operation at `point`.

            ValueError: If `op` cannot be placed on the circuit due to
                either an invalid location or gate radix mismatch.
        """
        self.pop(point)
        self.insert(point[0], op)

    def replace_gate(
        self,
        point: CircuitPointLike,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
        """Replace the operation at 'point' with `gate`."""
        self.replace(point, Operation(gate, location, params))

    def replace_with_circuit(
        self,
        point: CircuitPointLike,
        circuit: Circuit,
        location: Sequence[int],
    ) -> None:
        """Replace the operation at 'point' with `circuit`."""
        self.pop(point)
        self.insert_circuit(point[0], circuit, location)

    def fold(self, points: Sequence[CircuitPointLike]) -> None:
        """
        Fold the operations at the specified `points` into a CircuitGate.

        Args:
            points (Sequence[CircuitPointLike]): The positions of operations
                to group together.

        Raises:
            ValueError: If folding the specified points into a CircuitGate
                would change the result of `get_unitary`. This happens when
                their exists an operation within the bounds of `points`.

            IndexError: If any of `points` are invalid or out-of-range.
        """
        # Collect Bounds
        qudit_set = set()
        min_cycle_index = self.get_num_cycles()
        max_cycle_index = -1

        for point in points:
            if not self.is_point_in_range(point):
                raise IndexError('Out-of-range point.')

            qudit_set.add(point[1])
            cycle_index = point[0] if point[0] > 0 else self.get_num_cycles(
            ) + point[0]
            if cycle_index < min_cycle_index:
                min_cycle_index = point[0]
            if cycle_index > max_cycle_index:
                max_cycle_index = cycle_index

        # Collect Operations
        ops = list({self[point] for point in points})

        # Count Operations in bounds
        counting_ops = set()
        for cycle in self._circuit[min_cycle_index:max_cycle_index + 1]:
            for qudit_index in qudit_set:
                if cycle[qudit_index] is not None:
                    counting_ops.add(cycle[qudit_index])
        count = len(counting_ops)

        if len(ops) != count:
            raise ValueError(
                'Operations cannot be folded due to'
                ' another operation in the middle.',
            )

        # Pop, form CircuitGate, and insert in circuit
        circuit = self.batch_pop(points)
        circuit_gate = CircuitGate(circuit, True)
        qudits = sorted(list({point[1] for point in points}))
        self.insert_gate(min_cycle_index, circuit_gate, qudits)

    def copy(self) -> Circuit:
        """Return a deep copy of this circuit."""
        circuit = Circuit(self.get_size(), self.get_radixes())
        circuit._circuit = self._circuit.copy()
        circuit._gate_set = self._gate_set.copy()
        return circuit

    def get_slice(self, points: Sequence[CircuitPointLike]) -> Circuit:
        """Return a copy of a slice of this circuit."""
        qudits = sorted({point[1] for point in points})
        slice_size = len(qudits)
        slice_radixes = [self.get_radixes()[q] for q in qudits]
        slice = Circuit(slice_size, slice_radixes)
        for point in points:
            slice.append(self[point])
        return slice

    def clear(self) -> None:
        """Clear the circuit."""
        self._circuit = []
        self._gate_set = {}

    def operations(self, reversed: bool = False) -> Iterator[Operation]:
        return self.CircuitIterator(self._circuit, reversed)  # type: ignore

    def operations_with_points(
            self,
            reversed: bool = False,
    ) -> Iterator[tuple[CircuitPoint, Operation]]:
        return self.CircuitIterator(
            self._circuit, reversed, True,
        )  # type: ignore

    def operations_on_qudit(
        self,
        qudit_index: int,
        reversed: bool = False,
    ) -> Iterator[Operation]:
        return self.QuditIterator(
            qudit_index, self._circuit, reversed,
        )  # type: ignore

    def operations_on_qudit_with_points(
            self,
            qudit_index: int,
            reversed: bool = False,
    ) -> Iterator[tuple[CircuitPoint, Operation]]:
        return self.QuditIterator(
            qudit_index, self._circuit, reversed, True,
        )  # type: ignore

    # endregion

    # region Parameter Methods

    def get_param(self, param_index: int) -> float:
        """Return the parameter at param_index."""
        cycle, qudit, param = self.get_param_location(param_index)
        return self[cycle, qudit].params[param]

    def set_param(self, param_index: int, value: float) -> None:
        """Set a circuit parameter."""
        cycle, qudit, param = self.get_param_location(param_index)
        self[cycle, qudit].params[param] = value

    def freeze_param(self, param_index: int) -> None:
        """Freeze a circuit parameter to its current value."""
        cycle, qudit, param = self.get_param_location(param_index)
        op = self[cycle, qudit]
        gate = op.gate.with_frozen_params({param: op.params[param]})
        params = op.params.copy()
        params.pop(param)
        self.replace_gate((cycle, qudit), gate, op.location, params)

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

        Raises:
            IndexError: If the param_index is invalid.

        Examples:
            >>> circ = Circuit(1)
            >>> circ.append_gate(U3(), [0])
            >>> circ.append_gate(U3(), [0])
            >>> circ.get_num_params()
            6
            >>> circ.get_param_location(4)
            (1, 0, 1)
        """
        if param_index < 0:
            raise IndexError('Negative parameter index is not supported.')

        count = 0
        for point, op in self.operations_with_points():
            count += len(op.params)
            if count > param_index:
                param = param_index - (count - len(op.params))
                return (*point, param)

        raise IndexError('Out-of-range parameter index.')

    # endregion

    # region Circuit Logic Methods

    def get_inverse(self) -> Circuit:
        circuit = Circuit(self.get_size(), self.get_radixes())
        for op in reversed(self):
            circuit.append(
                Operation(
                    DaggerGate(op.gate),
                    op.location,
                    op.params,
                ),
            )
        return circuit

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
                'Expected sequence of integers'
                ', got %s' % type(qudit_permutation),
            )

        if len(qudit_permutation) != self.get_size():
            raise ValueError(
                'Expected qudit_permutation length equal to circuit size:'
                '%d, got %d' % (self.get_size(), len(qudit_permutation)),
            )

        if len(qudit_permutation) != len(set(qudit_permutation)):
            raise ValueError('Invalid permutation.')

        qudit_permutation = [int(q) for q in qudit_permutation]

        for op in self:
            op._location = [qudit_permutation[q] for q in op.location]

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
                utry.apply_right(op.get_unitary(gparams), op.location)
                param_index += op.get_num_params()
            else:
                utry.apply_right(op.get_unitary(), op.location)

        return utry.get_unitary()

    def get_statevector(self, in_state: StateVector) -> StateVector:
        pass  # TODO

    def optimize(self, method: str, **kwargs: Any) -> None:
        pass  # TODO

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
                right.apply_right(op.get_unitary(gparams), op.location)
                param_index += op.get_num_params()
            else:
                right.apply_right(op.get_unitary(), op.location)

        grads = []
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_left(
                    op.get_unitary(gparams),
                    op.location,
                    inverse=True,
                )
                param_index += op.get_num_params()
                grad = op.get_grad(gparams)
                grads.append(left.get_unitary() @ grad @ right.get_unitary())
                left.apply_right(op.get_unitary(gparams), op.location)
            else:
                right.apply_left(op.get_unitary(), op.location, inverse=True)
                grad = op.get_grad()
                grads.append(left.get_unitary() @ grad @ right.get_unitary())
                left.apply_right(op.get_unitary(gparams), op.location)

        return np.array(grads)

    def get_unitary_and_grad(
        self, params: Sequence[float] = [
        ],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """Return the unitary and gradient of the circuit."""
        if params:
            self.check_parameters(params)
            param_index = 0

        left = UnitaryBuilder(self.get_size(), self.get_radixes())
        right = UnitaryBuilder(self.get_size(), self.get_radixes())

        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_right(op.get_unitary(gparams), op.location)
                param_index += op.get_num_params()
            else:
                right.apply_right(op.get_unitary(), op.location)

        grads = []
        for op in self:
            if params:
                gparams = params[param_index:param_index + op.get_num_params()]
                right.apply_left(
                    op.get_unitary(gparams),
                    op.location,
                    inverse=True,
                )
                param_index += op.get_num_params()
                grad = op.get_grad(gparams)
                grads.append(left.get_unitary() @ grad @ right.get_unitary())
                left.apply_right(op.get_unitary(gparams), op.location)
            else:
                right.apply_left(op.get_unitary(), op.location, inverse=True)
                grad = op.get_grad()
                grads.append(left.get_unitary() @ grad @ right.get_unitary())
                left.apply_right(op.get_unitary(gparams), op.location)

        return left.get_unitary(), np.array(grads)

    # endregion

    # region Operation Collection Methods

    @overload
    def __getitem__(self, point: CircuitPointLike) -> Operation:
        ...

    @overload
    def __getitem__(self, points: Sequence[CircuitPoint] | slice) -> Circuit:
        ...

    def __getitem__(
        self,
        points: CircuitPointLike | Sequence[CircuitPoint] | slice,
    ) -> Operation | Circuit:
        """
        Retrieve an operation from a point or a circuit from a point sequence.

        Args:
            points (CircuitPoint | Sequence[CircuitPoint] | slice):
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
                return self.get_operation(points)  # type: ignore

        elif isinstance(points, slice) or is_sequence(points):
            return self.get_slice(points)  # type: ignore

        raise TypeError(
            'Invalid index type. Expected point'
            ', sequence of points, or slice'
            ', got %s' % type(points),
        )

    class CircuitIterator(
        Iterator[Union[Operation, tuple[CircuitPoint, Operation]]],
    ):
        def __init__(
                self,
                circuit: list[list[Operation | None]],
                reversed: bool = False,
                and_points: bool = False,
        ) -> None:
            self.circuit = circuit
            self.reversed = reversed
            self.and_points = and_points
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

        def __iter__(self) -> Iterator[
            Operation
            | tuple[CircuitPoint, Operation]
        ]:
            return self

        def __next__(self) -> Operation | tuple[CircuitPoint, Operation]:
            while self.dereference() is None:
                if not self.reversed:
                    self.increment_iter()
                else:
                    self.decrement_iter()
            op: Operation = self.dereference()  # type: ignore
            self.qudits_to_skip.update(op.location)
            if self.and_points:
                return CircuitPoint(self.cycle, self.qudit), op
            else:
                return op

    class QuditIterator(
            Iterator[Union[Operation, tuple[CircuitPoint, Operation]]],
    ):
        def __init__(
                self,
                qudit: int,
                circuit: list[list[Operation | None]],
                reversed: bool = False,
                and_points: bool = False,
        ) -> None:
            self.circuit = circuit
            self.reversed = reversed
            self.and_points = and_points
            self.qudit = qudit
            self.max_cycle = len(circuit)
            self.cycle = 0 if not reversed else self.max_cycle - 1

            self.max_qudit = 0 if self.max_cycle == 0 else len(circuit[0])
            if qudit >= self.max_qudit or qudit < 0:
                raise IndexError('Invalid qudit index for iterator.')

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

        def __iter__(self) -> Iterator[
            Operation
            | tuple[CircuitPoint, Operation]
        ]:
            return self

        def __next__(self) -> Operation | tuple[CircuitPoint, Operation]:
            while self.dereference() is None:
                if not self.reversed:
                    self.increment_iter()
                else:
                    self.decrement_iter()
            op: Operation = self.dereference()  # type: ignore
            if self.and_points:
                return CircuitPoint(self.cycle, self.qudit), op
            else:
                return op

    def __iter__(self) -> Iterator[Operation]:
        return self.CircuitIterator(self._circuit)  # type: ignore

    def __reversed__(self) -> Iterator[Operation]:
        return self.CircuitIterator(self._circuit, True)  # type: ignore

    def __contains__(self, op: object) -> bool:
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
            return False

    def __len__(self) -> int:
        """Return the number of operations in the circuit."""
        return self.get_num_operations()

    # endregion

    # region Operator Overloads

    def __invert__(self) -> Circuit:
        """Invert the circuit."""
        return self.get_inverse()

    def __eq__(self, rhs: object) -> bool:
        """
        Check for circuit equality.

        Two circuits are equal if: 1) They have the same number of
        operations. 2) All qudit radixes are equal. 3) All operations in
        simulation order are equal.
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

    def __ne__(self, rhs: object) -> bool:
        """Check for circuit inequality, see __eq__ for more info."""
        return not self == rhs

    def __add__(self, rhs: Circuit) -> Circuit:
        """Return a concatenated circuit copy."""
        circuit = Circuit(self.get_size(), self.get_radixes())
        circuit.append_circuit(self, list(range(self.get_size())))
        circuit.append_circuit(rhs, list(range(self.get_size())))
        return circuit

    def __mul__(self, rhs: int) -> Circuit:
        """Return a repeated circuit copy."""
        circuit = Circuit(self.get_size(), self.get_radixes())
        for x in range(rhs):
            circuit.append_circuit(self, list(range(self.get_size())))
        return circuit

    def __radd__(self, lhs: Circuit) -> Circuit:
        """Return a concatenated circuit copy."""
        circuit = Circuit(self.get_size(), self.get_radixes())
        circuit.append_circuit(lhs, list(range(self.get_size())))
        circuit.append_circuit(self, list(range(self.get_size())))
        return circuit

    def __iadd__(self, rhs: Circuit) -> None:
        """Return a concatenated circuit copy."""
        self.append_circuit(rhs, list(range(self.get_size())))

    def __imul__(self, rhs: int) -> None:
        """Return a repeated circuit copy."""
        circuit = self.copy()
        for x in range(rhs - 1):
            self.append_circuit(circuit, list(range(self.get_size())))

    # endregion

    # region IO Methods

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    def save(self, filename: str) -> None:
        pass

    @ staticmethod
    def from_file(filename: str) -> Circuit:
        pass

    @ staticmethod
    def from_str(str: str) -> Circuit:
        pass

    @ staticmethod
    def from_unitary(utry: UnitaryMatrix) -> Circuit:
        circuit = Circuit(utry.get_size(), utry.get_radixes())
        circuit.append_gate(
            ConstantUnitaryGate(utry), list(
                range(
                    utry.get_size(),
                ),
            ),
        )
        return circuit

    # endregion

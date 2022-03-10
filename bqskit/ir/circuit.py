"""This module implements the Circuit class."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import List
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.composed.tagged import TaggedGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.iterator import CircuitIterator
from bqskit.ir.lang import get_language
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtCost
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.instantiaters import instantiater_order
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion
from bqskit.ir.region import CircuitRegionLike
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.state.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.statemap import StateVectorMap
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.random import seed_random_sources
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_sequence_of_int
from bqskit.utils.typing import is_square_matrix
from bqskit.utils.typing import is_valid_radixes
from bqskit.utils.typing import is_vector

if TYPE_CHECKING:
    from bqskit.ir.opt.cost.function import CostFunction

_logger = logging.getLogger(__name__)


class Circuit(DifferentiableUnitary, StateVectorMap, Collection[Operation]):
    """
    Circuit class.

    A Circuit is a quantum program composed of operation objects.

    The operations are organized in 2-dimensions, and are indexed by
    a CircuitPoint. The first index corresponds to an operation's cycle.
    This describes when an operation will be executed. The second index
    is the qudit index and describes where the operation will execute.
    An operation can be multi-qudit meaning its location - list of qudit
    indices describing which qudits the operation affects - contains
    multiple indices. All operations exist in only one cycle at a time,
    but if an operation is multi-qudit, then it is pointed to by multiple
    qudit indices.

    Invariants:
        1. A circuit method will never complete with an idle cycle.
           An idle cycle is one that contains no gates.

        2. No one logical operation will ever be pointed to from more
           than one cycle.

        3. Iterating through the entire circuit always produces
           operations in simulation order. This means that if those
           operations were applied to a quantum state in the same
           order, then the result is the same as simulating the circuit.

    Notes:
        While a guarantee is made that the circuit never has any idle
        cycles, this means that cycles can be deleted or inserted
        during a method call. Therefore, cycle indices may need to be
        updated in between calls to circuit methods. There are several
        "batch" variants of methods that can handle this for you.
    """

    def __init__(self, num_qudits: int, radixes: Sequence[int] = []) -> None:
        """
        Build an empty circuit with the specified number of qudits.

        By default, all qudits are qubits, but this can be changed
        with radixes.

        Args:
            num_qudits (int): The number of qudits in this circuit.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: if `num_qudits` is nonpositive.

        Examples:
            >>> circ = Circuit(4)  # Creates four-qubit empty circuit.

            >>> circ = Circuit(2, [2, 3])  # Creates one qubit and one qutrit.

            >>> circ = Circuit(2)
            >>> circ.append_gate(HGate(), 0)
            >>> circ.append_gate(CXGate(), (0, 1))
            >>> circ.append_gate(HGate(), 1)
            >>> circ.get_unitary()
            ... array([[ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j],
            ...        [ 0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j],
            ...        [ 0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j],
            ...        [-0.5+0.j,  0.5+0.j,  0.5+0.j, -0.5+0.j]])
            >>> circ.get_statevector([1, 0, 0, 0])
            ... array([ 0.5+0.j,  0.5+0.j,  0.5+0.j, -0.5+0.j])
        """

        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected integer num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError(
                f'Expected positive integer for num_qudits, got {num_qudits}.',
            )

        self._num_qudits = int(num_qudits)
        self._radixes = tuple(
            radixes if len(radixes) > 0 else [2] * self.num_qudits,
        )

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(self.radixes) != self.num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(self.radixes), self.num_qudits),
            )

        self._circuit: list[list[Operation | None]] = []
        self._gate_info: dict[Gate, int] = {}

    # region Circuit Properties

    @property
    def num_params(self) -> int:
        """The total number of parameters in the circuit."""
        num_params_acm = 0
        for gate, count in self._gate_info.items():
            num_params_acm += gate.num_params * count
        return num_params_acm

    @property
    def num_operations(self) -> int:
        """The total number of operations in the circuit."""
        num_gates_acm = 0
        for _, count in self._gate_info.items():
            num_gates_acm += count
        return num_gates_acm

    @property
    def num_cycles(self) -> int:
        """The number of cycles in the circuit."""
        return len(self._circuit)

    @property
    def params(self) -> npt.NDArray[np.float64]:
        """The stored parameters for the circuit."""
        return np.array(sum((list(op.params) for op in self), []))

    @property
    def depth(self) -> int:
        """The length of the critical path in the circuit."""
        qudit_depths = np.zeros(self.num_qudits, dtype=int)
        for op in self:
            new_depth = max(qudit_depths[list(op.location)]) + 1
            qudit_depths[list(op.location)] = new_depth
        return int(max(qudit_depths))

    @property
    def parallelism(self) -> float:
        """The amount of parallelism in the circuit."""
        depth = self.depth

        if depth == 0:
            return 0

        weighted_num_operations = np.sum([
            gate.num_qudits * count
            for gate, count in self._gate_info.items()
        ])

        return float(weighted_num_operations / depth)

    @property
    def coupling_graph(self) -> set[tuple[int, int]]:
        """
        The qudit connectivity in the circuit.

        Returns:
            set[tuple[int, int]]: The coupling graph required by
            the circuit. The graph is returned as an edge list.

        Notes:
            - Multi-qudit gates set participating qudits to have
              all-to-all connectivity.

            - The graph is undirected.
        """
        coupling_graph = set()
        for op in self:
            for q1 in op.location:
                for q2 in op.location:
                    if q1 == q2:
                        continue
                    coupling_graph.add((min(q1, q2), max(q1, q2)))
        return coupling_graph

    @property
    def gate_set(self) -> set[Gate]:
        """The set of gates in the circuit."""
        return set(self._gate_info.keys())

    @property
    def active_qudits(self) -> list[int]:
        """The qudits involved in at least one operation."""
        return [
            qudit
            for qudit in range(self.num_qudits)
            if not self.is_qudit_idle(qudit)
        ]

    def is_differentiable(self) -> bool:
        """Check if all gates are differentiable."""
        return all(
            isinstance(gate, DifferentiableUnitary)
            for gate in self.gate_set
        )

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

        if not is_integer(radix):
            raise TypeError('Expected integer for radix, got: %s', type(radix))

        if radix < 2:
            raise ValueError('Expected radix to be >= 2, got %d' % radix)

        self._num_qudits += 1
        self._radixes = self.radixes + (radix,)

        for cycle in self._circuit:
            cycle.append(None)

    def extend_qudits(self, radixes: Iterable[int]) -> None:
        """
        Append many qudits to the circuit.

        Args:
            radixes (Iterable[int]): The radix for each qudit to append.

        Raises:
            ValueError: If any radix in `radixes` is < 2.
        """

        for radix in radixes:
            self.append_qudit(radix)

    def insert_qudit(self, qudit_index: int, radix: int = 2) -> None:
        """
        Insert a qudit into the circuit.

        Args:
            qudit_index (int): The index where to insert the qudit.

            radix (int): The radix of the qudit. (Default: qubit)

        Raises:
            ValueError: If `radix` is < 2.
        """

        if not is_integer(qudit_index):
            raise TypeError(
                f'Expected integer for qudit_index, got: {type(qudit_index)}',
            )

        if not is_integer(radix):
            raise TypeError(f'Expected integer for radix, got: {type(radix)}')

        if radix < 2:
            raise ValueError(f'Expected radix to be >= 2, got {radix}')

        if qudit_index >= self.num_qudits:
            return self.append_qudit(radix)

        if qudit_index <= -self.num_qudits:
            qudit_index = 0
        elif qudit_index < 0:
            qudit_index = self.num_qudits + qudit_index

        # Update circuit properties
        self._num_qudits += 1
        radix_list = list(self.radixes)
        radix_list.insert(qudit_index, radix)
        self._radixes = tuple(radix_list)

        # Insert qudit
        for cycle in self._circuit:
            cycle.insert(qudit_index, None)

            # Renumber gates with now-invalid locations
            qudits_to_skip: list[int] = []
            for i, op in enumerate(cycle[qudit_index:]):
                if op is None or i + qudit_index in qudits_to_skip:
                    continue
                op._location = CircuitLocation([
                    index if index < qudit_index else index + 1
                    for index in op.location
                ])
                qudits_to_skip.extend(op.location)

    def pop_qudit(self, qudit_index: int) -> None:
        """
        Pop a qudit from the circuit and all gates attached to it.

        Args:
            qudit_index (int): The index of the qudit to pop.

        Raises:
            IndexError: If `qudit_index` is out of range.

            ValueError: If circuit only has one qudit.
        """

        if not is_integer(qudit_index):
            raise TypeError(
                f'Expected integer for qudit_index, got: {type(qudit_index)}',
            )

        if not self.is_qudit_in_range(qudit_index):
            raise IndexError(f'Qudit index ({qudit_index}) is out-of-range.')

        if self.num_qudits == 1:
            raise ValueError('Cannot pop only qudit in circuit.')

        if qudit_index < 0:
            qudit_index = self.num_qudits + qudit_index

        # Remove gates attached to popped qudit
        points = []
        for cycle_index, cycle in enumerate(self._circuit):
            if cycle[qudit_index] is not None:
                points.append((cycle_index, qudit_index))
        self.batch_pop(points)

        # Update circuit properties
        self._num_qudits -= 1
        radix_list = list(self.radixes)
        radix_list.pop(qudit_index)
        self._radixes = tuple(radix_list)

        # Remove qudit
        for cycle_index, cycle in enumerate(self._circuit):
            cycle.pop(qudit_index)

        # Renumber gates with now-invalid locations
        for cycle_index, cycle in enumerate(self._circuit):
            qudits_to_skip: list[int] = []
            for i, op in enumerate(cycle[qudit_index:]):
                if op is None or i + qudit_index in qudits_to_skip:
                    continue
                op._location = CircuitLocation([
                    index if index < qudit_index else index - 1
                    for index in op.location
                ])
                qudits_to_skip.extend(op.location)

    def is_qudit_in_range(self, qudit_index: int) -> bool:
        """Return true if `qudit_index` is in-range for the circuit."""

        if not is_integer(qudit_index):
            raise TypeError(
                f'Expected integer for qudit_index, got: {type(qudit_index)}',
            )

        return (
            qudit_index < self.num_qudits
            and qudit_index >= -self.num_qudits
        )

    def is_qudit_idle(self, qudit_index: int) -> bool:
        """Return true if the qudit is not involved in any operations."""

        if not is_integer(qudit_index):
            raise TypeError(
                f'Expected integer for qudit_index, got: {type(qudit_index)}',
            )

        return all(cycle[qudit_index] is None for cycle in self._circuit)

    # endregion

    # region Cycle Methods

    def _append_cycle(self) -> None:
        """Appends an idle cycle to the end of the circuit."""
        self._circuit.append([None] * self.num_qudits)

    def _insert_cycle(self, cycle_index: int) -> None:
        """Inserts an idle cycle in the circuit."""
        self._circuit.insert(cycle_index, [None] * self.num_qudits)

    def pop_cycle(self, cycle_index: int) -> None:
        """
        Pop a cycle from the circuit and all operations in it.

        Args:
            cycle_index (int): The index of the cycle to pop.

        Raises:
            IndexError: If `cycle_index` is out of range.
        """

        if not is_integer(cycle_index):
            raise TypeError(
                f'Expected integer for cycle_index, got: {type(cycle_index)}',
            )

        if not self.is_cycle_in_range(cycle_index):
            raise IndexError(f'Cycle index ({cycle_index}) is out-of-range.')

        qudits_to_skip: list[int] = []
        for qudit_index, op in enumerate(self._circuit[cycle_index]):
            if op is not None and qudit_index not in qudits_to_skip:
                qudits_to_skip.extend(op.location)
                self._gate_info[op.gate] -= 1
                if self._gate_info[op.gate] <= 0:
                    del self._gate_info[op.gate]

        self._circuit.pop(cycle_index)

    def _is_cycle_idle(self, cycle_index: int) -> bool:
        """Return true if the cycle is idle, that is it contains no gates."""
        return all(op is None for op in self._circuit[cycle_index])

    def is_cycle_in_range(self, cycle_index: int) -> bool:
        """Return true if cycle is a valid in-range index in the circuit."""

        if not is_integer(cycle_index):
            raise TypeError(
                f'Expected integer for cycle_index, got: {type(cycle_index)}',
            )

        return (
            cycle_index < self.num_cycles
            and cycle_index >= -self.num_cycles
        )

    def is_cycle_unoccupied(
        self,
        cycle_index: int,
        location: CircuitLocationLike,
    ) -> bool:
        """
        Check if a cycle is unoccupied for all qudits in `location`.

        Args:
            cycle_index (int): The cycle to check.

            location (CircuitLocationLike): The set of qudits to check.

        Returns:
            bool: True if the `cycle_index` at `location` is unoccupied.

        Raises:
            IndexError: If `cycle_index` is out of range.

        Examples:
            >>> circuit = Circuit(2)
            >>> circuit.append_gate(HGate(), [0])
            >>> circuit.append_gate(XGate(), [0])
            >>> circuit.append_gate(ZGate(), [1])
            >>> circuit.is_cycle_unoccupied(0, [0])
            False
            >>> circuit.is_cycle_unoccupied(1, [1])
            True
        """
        if not is_integer(cycle_index):
            raise TypeError(
                'Expected integer for cycle_index, got: %s', type(cycle_index),
            )

        if not CircuitLocation.is_location(location):
            raise TypeError('Invalid location.')

        if not self.is_cycle_in_range(cycle_index):
            raise IndexError('Out-of-range cycle index: %d.' % cycle_index)

        location = CircuitLocation(location)

        if max(location) > self.num_qudits:
            raise ValueError('Location has an out-of-range qudit index.')

        for qudit_index in location:
            if self._circuit[cycle_index][qudit_index] is not None:
                return False

        return True

    def find_available_cycle(self, location: CircuitLocationLike) -> int:
        """
        Finds the first available cycle for qudits in `location`.

        An available cycle for `location` is one where it and all
        cycles after it are unoccupied for `location`.

        Args:
            localtion (CircuitLocationLike): Find a cycle for this location.

        Returns:
            int: The first available cycle.

        Raises:
            ValueError: If no available cycle exists.

        Examples:
            >>> circuit = Circuit(2)
            >>> circuit.append_gate(HGate(), [0])
            >>> circuit.find_available_cycle([1])
            0
            >>> circuit.append_gate(XGate(), [0])
            >>> circuit.append_gate(ZGate(), [1])
            >>> circuit.find_available_cycle([1])
            1
        """

        location = CircuitLocation(location)

        if max(location) > self.num_qudits:
            raise ValueError('Location has an out-of-range qudit index.')

        # No available cycle
        if self.num_cycles == 0:
            return -1

        # Iterate through cycles in reverse order
        for cycle in range(self.num_cycles - 1, -1, -1):
            # Find the first occupied cycle
            if not self.is_cycle_unoccupied(cycle, location):
                # The first available cycle is the previous one
                if cycle == self.num_cycles - 1:
                    return -1
                return cycle + 1

        # If we didn't find an occupied cycle,
        # then they are all unoccupied.
        return 0

    def _find_available_or_append_cycle(
        self,
        location: CircuitLocationLike,
    ) -> int:
        """Find the first available cycle, if none exists append one."""

        available_cycle = self.find_available_cycle(location)

        # If no available cycle
        if available_cycle == -1:
            self._append_cycle()
            return self.num_cycles - 1

        return available_cycle

    # endregion

    # region Point Methods

    def is_point_in_range(self, point: CircuitPointLike) -> bool:
        """Return true if `point` is a valid in-range index in the circuit."""
        if not CircuitPoint.is_point(point):
            raise TypeError(f'Expected CircuitPoint, got: {type(point)}.')

        return (
            self.is_cycle_in_range(point[0])
            and self.is_qudit_in_range(point[1])
        )

    def is_point_idle(self, point: CircuitPointLike) -> bool:
        """Return true if an operation does not exist at `point`."""
        if not CircuitPoint.is_point(point):
            raise TypeError(f'Expected CircuitPoint, got {type(point)}.')

        return self._circuit[point[0]][point[1]] is None

    def normalize_point(self, point: CircuitPointLike) -> CircuitPoint:
        """Flip negative indices to positive indices."""
        if not self.is_point_in_range(point):
            raise IndexError('Out-of-range point.')

        cycle = point[0] if point[0] >= 0 else self.num_cycles + point[0]
        qudit = point[1] if point[1] >= 0 else self.num_qudits + point[1]
        return CircuitPoint(cycle, qudit)

    # endregion

    # region Operation/Gate/Circuit Methods

    def check_valid_operation(self, op: Operation) -> None:
        """Check that `op` can be applied to the circuit."""
        if not isinstance(op, Operation):
            raise TypeError('Expected Operation got %s.' % type(op))

        if not all([qudit < self.num_qudits for qudit in op.location]):
            raise ValueError('Operation location mismatch with Circuit.')

        for op_radix, circ_radix_idx in zip(op.radixes, op.location):
            if op_radix != self.radixes[circ_radix_idx]:
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
            Operation: The operation at `point`.

        Examples:
            >>> circuit = Circuit(2)
            >>> circuit.append_gate(HGate(), [0])
            >>> circuit.append_gate(CNOTGate(), [0, 1])
            >>> circuit.get_operation((1, 0))
            ... CNOTGate()@(0, 1)
        """
        if not self.is_point_in_range(point):
            raise IndexError('Out-of-range or invalid point.')

        op = self._circuit[point[0]][point[1]]

        if op is None:
            raise IndexError(
                'No operation exists at the specified point: %s.' % str(point),
            )

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
                Inclusive. (Default: End of the circuit.)

        Returns:
            CircuitPoint: The first point that contains `op`.

        Raises:
            ValueError: If `op` is not found.

            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.

        Examples:
            >>> circuit = Circuit(1)
            >>> opH = Operation(HGate(), [0])
            >>> circuit.append(opH)
            >>> circuit.point(opH)
            (0, 0)
            >>> opX = Operation(XGate(), [0])
            >>> circuit.point(opX)
            (1, 0)
        """

        if not isinstance(op, (Operation, Gate)):
            raise TypeError(f'Expected gate or operation, got {type(op)}.')

        end = end if end is not None else (-1, -1)
        start = self.normalize_point(start)
        end = self.normalize_point(end)

        if isinstance(op, Operation):
            self.check_valid_operation(op)
            if op.gate not in self._gate_info:
                raise ValueError('No such operation exists in the circuit.')

        elif isinstance(op, Gate):
            if op not in self._gate_info:
                raise ValueError('No such operation exists in the circuit.')

        if not self.is_point_in_range(start):
            raise IndexError('Out-of-range or invalid start point.')

        if not self.is_point_in_range(end):
            raise IndexError('Out-of-range or invalid end point.')

        if isinstance(op, Operation):
            qudit_index = op.location[0]
            for i, cycle in enumerate(self._circuit[start[0]:end[0] + 1]):
                if cycle[qudit_index] is not None and cycle[qudit_index] == op:
                    return CircuitPoint(start[0] + i, qudit_index)
        else:
            for i, cycle in enumerate(self._circuit[start[0]:end[0] + 1]):
                for q, _op in enumerate(cycle[start[1]:end[1] + 1]):
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
            `circuit.append(op)` does not imply `op` is last in simulation
            order but it implies `op` is in the last cycle of circuit.

        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op) # Appends a Hadamard gate to qudit 0.
        """
        self.check_valid_operation(op)

        if op.gate not in self._gate_info:
            self._gate_info[op.gate] = 0
        self._gate_info[op.gate] += 1

        cycle_index = self._find_available_or_append_cycle(op.location)

        for qudit_index in op.location:
            self._circuit[cycle_index][qudit_index] = op

    def append_gate(
        self,
        gate: Gate,
        location: CircuitLocationLike,
        params: RealVector = [],
    ) -> None:
        """
        Append the gate object to the circuit on the qudits in location.

        Args:
            gate (Gate): The gate to append.

            location (CircuitLocationLike): Apply the gate to these qudits.

            params (RealVector): The gate's parameters.
                (Default: all zeros)

        Examples:
            >>> circ = Circuit(1)
            >>> # Append a Hadamard gate to qudit 0.
            >>> circ.append_gate(H(), [0])

        See Also:
            :func:`append`
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate, got %s.' % type(gate))

        self.append(Operation(gate, location, params))

    def append_circuit(
        self,
        circuit: Circuit,
        location: CircuitLocationLike,
    ) -> None:
        """
        Append `circuit` at the qudit location specified.

        Args:
            circuit (Circuit): The circuit to append.

            location (CircuitLocationLike): Apply the circuit to these qudits.

        Raises:
            ValueError: If `circuit` is not the same size as `location`.

        See Also:
            :func:`append`
        """
        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if not CircuitLocation.is_location(location):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)

        if circuit.num_qudits != len(location):
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

        if self.num_cycles == 0:
            self._append_cycle()

        if not self.is_cycle_in_range(cycle_index):
            if cycle_index < -self.num_cycles:
                cycle_index = 0
            else:
                cycle_index = self.num_cycles
                self._append_cycle()

        if op.gate not in self._gate_info:
            self._gate_info[op.gate] = 0
        self._gate_info[op.gate] += 1

        if not self.is_cycle_unoccupied(cycle_index, op.location):
            self._insert_cycle(cycle_index)
            cycle_index -= 1 if cycle_index < 0 else 0

        for qudit_index in op.location:
            self._circuit[cycle_index][qudit_index] = op

    def insert_gate(
        self,
        cycle_index: int,
        gate: Gate,
        location: CircuitLocationLike,
        params: RealVector = [],
    ) -> None:
        """
        Insert the gate object in the circuit on the qudits in location.

        After this, if cycle was in range, you can expect:
        `all([self[cycle_index, idx].gate == gate for idx in location])`

        Args:
            cycle_index (int): The cycle to insert the gate.

            gate (Gate): The gate to insert.

            location (CircuitLocationLike): Apply the gate to this set of
                qudits.

            params (RealVector): The gate's parameters.

        Raises:
            IndexError: If the specified cycle doesn't exist.

            ValueError: If `gate` cannot be placed on the circuit due to
                either an invalid location or gate radix mismatch.

        See Also:
            :func:`insert`
        """
        self.insert(cycle_index, Operation(gate, location, params))

    def insert_circuit(
        self,
        cycle_index: int,
        circuit: Circuit,
        location: CircuitLocationLike,
    ) -> None:
        """
        Insert `circuit` at the cycle and location specified.

        Args:
            cycle_index (int): The cycle to insert the circuit.

            circuit (Circuit): The circuit to insert.

            location (CircuitLocationLike): Apply the circuit to these
                qudits.

        Raises:
            ValueError: If `circuit` is not the same size as `location`.

        See Also:
            :func:`insert`
        """

        if not is_integer(cycle_index):
            raise TypeError(
                f'Expected integer cycle index, got: {cycle_index}.',
            )

        if not CircuitLocation.is_location(location):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)

        if circuit.num_qudits != len(location):
            raise ValueError('Circuit and location size mismatch.')

        for op in reversed(circuit):
            mapped_location = [location[q] for q in op.location]
            self.insert(
                cycle_index,
                Operation(op.gate, mapped_location, op.params),
            )

    def remove(self, op: Operation | Gate) -> None:
        """
        Removes the first occurrence of `op` in the circuit.

        Args:
            op (Operation | Gate): The Operation or Gate to remove.

        Raises:
            ValueError: If the `op` doesn't exist in the circuit.

            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.

        Examples:
            >>> circ = Circuit(1)
            >>> op = Operation(H(), [0])
            >>> circ.append(op)
            >>> circ.num_operations
            1
            >>> circ.remove(op)
            >>> circ.num_operations
            0

        See Also:
            :func:`remove_all`
        """
        self.pop(self.point(op))

    def remove_all(self, op: Operation | Gate) -> None:
        """
        Removes the all occurrences of `op` in the circuit.

        Args:
            op (Operation | Gate): The Operation or Gate to remove.

        Raises:
            ValueError: If the `op` doesn't exist in the circuit.

            ValueError: If `op` could not have been placed on the circuit
                due to either an invalid location or gate radix mismatch.

        See Also:
            :func:`remove`
        """
        while True:
            try:
                self.pop(self.point(op))
            except (ValueError, IndexError):
                break

    def count(self, op: Operation | Gate) -> int:
        """
        Count the number of times `op` occurs in the circuit.

        Args:
            op (Operation | Gate): The operation or gate to count.

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
            if op.gate not in self._gate_info:
                return 0

            qudit_index = op.location[0]
            for _op in self.operations(qudits_or_region=[qudit_index]):
                if _op == op:
                    count += 1
        elif isinstance(op, Gate):
            if op not in self._gate_info:
                return 0
            return self._gate_info[op]
        else:
            raise TypeError('Expected gate or operation, got %s.' % type(op))

        return count

    def pop(self, point: CircuitPointLike | None = None) -> Operation:
        """
        Pop the operation at `point`, defaults to last operation.

        Args:
            point (CircuitPointLike | None): The cycle and qudit index
                to pop from.

        Returns:
            Operation: The popped operation is returned.

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
            cycle_index = self.num_cycles - 1
            for i, op in enumerate(reversed(self._circuit[cycle_index])):
                if op is not None:
                    point = (cycle_index, self.num_qudits - 1 - i)
                    break

        if point is None:
            raise IndexError('Pop from empty circuit.')

        op = self[point]

        for qudit_index in op.location:
            self._circuit[point[0]][qudit_index] = None

        self._gate_info[op.gate] -= 1
        if self._gate_info[op.gate] <= 0:
            del self._gate_info[op.gate]

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
            Circuit: The circuit formed from all the popped operations.

        Raises:
            IndexError: If any of `points` are out-of-range.

            IndexError: If all of `points` are invalid.
        """
        if not all(self.is_point_in_range(point) for point in points):
            raise IndexError('Out-of-range point.')

        # Sort points
        points = sorted(points)

        # Collect operations avoiding duplicates
        points_to_skip: list[CircuitPointLike] = []
        ops_and_cycles: list[tuple[Operation, int]] = []
        for point in points:
            if point in points_to_skip:
                continue
            if self.is_point_idle(point):
                continue

            op = self[point]
            ops_and_cycles.append((op, point[0]))
            points_to_skip.extend((point[0], q) for q in op.location)

        ops = [op for op, _ in ops_and_cycles]
        points = [(cycle, op.location[0]) for op, cycle in ops_and_cycles]

        if len(points) == 0:
            raise IndexError('No operations exists at any of the points.')

        # Pop gates, tracking if the circuit popped a cycle
        num_cycles = self.num_cycles
        for i, point in enumerate(points):

            try:
                self.pop(point)
            except IndexError:  # Silently discard multi-points-one-op errors
                pass

            if num_cycles != self.num_cycles:
                num_cycles = self.num_cycles
                points[i + 1:] = [(point[0] - 1, point[1])
                                  for point in points[i + 1:]]

        # Form new circuit and return
        qudits = list(set(sum((tuple(op.location) for op in ops), ())))
        qudits = sorted(qudits)
        radixes = [self.radixes[q] for q in qudits]
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

            ValueError: If `point.qudit` is not in `op.location`
        """
        if len(self[point].location.intersection(op.location)) == 0:
            raise ValueError("Point's qudit is not in operation's location.")

        self.pop(point)
        self.insert(point[0], op)

    def batch_replace(
        self,
        points: Sequence[CircuitPointLike],
        ops: Sequence[Operation],
    ) -> None:
        """
        Replace the operations at `points` with `ops`.

        Args:
            points (Sequence[CircuitPointLike]): The indices of the
                operations to replace.

            ops (Sequence[Operation]): The to-be-inserted operations.

        Raises:
            IndexError: If there is no operation at some point.

            ValueError: If any op cannot be placed on the circuit due to
                either an invalid location or gate radix mismatch.

            ValueError: If any `point.qudit` is not in `op.location`

            ValueError: If `points` and `ops` have different lengths.
        """
        if len(points) != len(ops):
            raise ValueError('Points and Ops have different lengths.')

        points_and_ops = sorted(zip(points, ops), key=lambda x: x[0][0])
        num_cycles = self.num_cycles

        for point, op in points_and_ops:
            shrink_amount = num_cycles - self.num_cycles
            shifted_point = (point[0] - shrink_amount, point[1])
            self.replace(shifted_point, op)

    def replace_gate(
        self,
        point: CircuitPointLike,
        gate: Gate,
        location: CircuitLocationLike,
        params: RealVector = [],
    ) -> None:
        """Replace the operation at 'point' with `gate`."""
        self.replace(point, Operation(gate, location, params))

    def replace_with_circuit(
        self,
        point: CircuitPointLike,
        circuit: Circuit,
    ) -> None:
        """Replace the operation at 'point' with `circuit`."""
        op = self.pop(point)

        if circuit.num_qudits != op.num_qudits:
            raise ValueError('Cannot replace operation with circuit.')

        if circuit.radixes != tuple(self.radixes[x] for x in op.location):
            raise ValueError('Cannot replace operation with circuit.')

        self.insert_circuit(point[0], circuit, op.location)

    def copy(self) -> Circuit:
        """Return a deep copy of this circuit."""
        circuit = Circuit(self.num_qudits, self.radixes)
        circuit._circuit = copy.deepcopy(self._circuit)
        circuit._gate_info = copy.deepcopy(self._gate_info)
        return circuit

    def become(self, circuit: Circuit, deepcopy: bool = True) -> None:
        """Become a copy of `circuit`."""
        if deepcopy:
            self._num_qudits = circuit.num_qudits
            self._radixes = circuit.radixes
            self._circuit = copy.deepcopy(circuit._circuit)
            self._gate_info = copy.deepcopy(circuit._gate_info)
        else:
            self._num_qudits = circuit.num_qudits
            self._radixes = circuit.radixes
            self._circuit = copy.copy(circuit._circuit)
            self._gate_info = copy.copy(circuit._gate_info)

    def clear(self) -> None:
        """Clear the circuit."""
        self._circuit = []
        self._gate_info = {}

    # endregion

    # region Region Methods

    def is_valid_region(
        self,
        region: CircuitRegionLike,
        strict: bool = False,
    ) -> bool:
        """Return true if `region` is valid in the context of this circuit."""
        try:
            self.check_region(region, strict)
        except ValueError:
            return False
        return True

    def check_region(
        self,
        region: CircuitRegionLike,
        strict: bool = False,
    ) -> None:
        """
        Check `region` to be a valid in the context of this circuit.

        Args:
            region (CircuitRegionLike): The region to check.

            strict (bool): If True, fail on any disconnect, even if there
                are no gates in the disconnect. (Default: False)

        Raises:
            ValueError: If `region` includes qudits not in the circuit,
                or if the region is too large for the circuit.

            ValueError: If the region cannot be folded due to a possible
                change to simulation order. This can happen when the
                region is disconnected between qudits and outside gates
                occur in the disconnect.
        """
        if not CircuitRegion.is_region(region):
            raise TypeError(f'Expected a CircuitRegion, got: {type(region)}.')

        region = CircuitRegion(region)

        if not CircuitLocation.is_location(region.location, self.num_qudits):
            raise ValueError('Region circuit location mismatch.')

        if region.max_cycle >= self.num_cycles:
            raise ValueError(
                'Region goes off circuit; '
                f'circuit only has {self.num_cycles} cycles, '
                f"but region's maximum cycle is {region.max_cycle}.",
            )

        for qudit_index, cycle_intervals in region.items():
            for other_qudit_index, other_cycle_intervals in region.items():
                if cycle_intervals.overlaps(other_cycle_intervals):
                    continue
                involved_qudits = {qudit_index}
                min_index = min(
                    cycle_intervals.upper,
                    other_cycle_intervals.upper,
                )
                max_index = max(
                    cycle_intervals.lower,
                    other_cycle_intervals.lower,
                )
                for cycle_index in range(min_index + 1, max_index):
                    try:
                        ops = self[cycle_index, involved_qudits]
                    except IndexError:
                        continue

                    if strict:
                        raise ValueError('Disconnect detected in region.')

                    if any(other_qudit_index in op.location for op in ops):
                        raise ValueError(
                            'Disconnected region has excluded gate in middle.',
                        )

                    for op in ops:
                        involved_qudits.update(op.location)

    def straighten(
        self,
        region: CircuitRegionLike,
    ) -> tuple[CircuitRegion, int, CircuitRegion]:
        """
        Push gates back so the region has a single starting cycle.

        Args:
            region (CircuitRegionLike): The region to straighten.

        Returns:
            (tuple[CircuitRegion, int, CircuitRegion]):
                Statistics on how gates where moved to straighten this
                region. The first return value is the straightened region.
                The second integer return value is the net number of
                new cycles inserted at `region.min_cycle`. The last
                return value is the shadow region, or the portion of the
                middle region which did not move. See the Notes for
                more info.

        Raises:
            ValueError: If the region is invalid, see `circuit.check_region`.

        Notes:
            When isolating a region, the circuit is divided into three
            parts by `region.min_cycle` and `region.max_min_cycle`. The
            left region is left unchanged and the right region
            potentially moves to the right. Gates in the middle can
            either be moved to the right or left unchanged depending
            on their location. The last return value describes the
            cycle-qudits that did not move from the middle portion.
        """
        if len(region) == 0:
            return CircuitRegion({}), 0, CircuitRegion({})

        self.check_region(region)
        region = self.downsize_region(region)

        # Add idle cycles to create space
        shadow_length = region.max_min_cycle - region.min_cycle
        shadow_start = region.min_cycle
        for i in range(shadow_length):
            self._insert_cycle(shadow_start)
        region = region.shift_right(shadow_length)

        # Track region shadow and move gates
        idle_cycles: list[int] = []
        shadow_qudits: set[int] = set(region.keys())
        # shadow_map = {qudit: region[qudit].lower for qudit in shadow_qudits}
        shadow_map = {
            qudit: min(
                region.min_cycle - 1,
                region[qudit].upper - shadow_length,
            )
            for qudit in shadow_qudits
        }

        for i in range(shadow_length):
            old_cycle_index = region.max_min_cycle - 1 - i
            new_cycle_index = region.min_cycle - 1 - i
            gate_moved = False
            qudits_to_add_to_shadow: list[int] = []

            for qudit_index in shadow_qudits:
                if (
                    qudit_index not in region
                    or old_cycle_index < region[qudit_index][0]
                ):
                    op = self._circuit[old_cycle_index][qudit_index]
                    if op is not None:
                        gate_moved = True
                        for qudit in op.location:
                            self._circuit[new_cycle_index][qudit] = op
                            self._circuit[old_cycle_index][qudit] = None
                        qudits_to_add_to_shadow.extend(op.location)

            shadow_qudits.update(qudits_to_add_to_shadow)
            for qudit in shadow_qudits:
                if qudit not in shadow_map:
                    shadow_map[qudit] = new_cycle_index

            if not gate_moved:
                idle_cycles.append(new_cycle_index)

        for i, cycle_index in enumerate(sorted(idle_cycles)):
            self.pop_cycle(cycle_index - i)

        region = region.shift_left(len(idle_cycles))

        # Prep output
        region = CircuitRegion({
            qudit_index: (region.min_cycle, region[qudit_index][1])
            for qudit_index in region
        })
        net_new_cycles = shadow_length - len(idle_cycles)
        shadow_region = CircuitRegion({
            qudit_index: (shadow_start, shadow_map[qudit_index])
            for qudit_index in shadow_qudits
            if shadow_start <= shadow_map[qudit_index]
        })
        return region, net_new_cycles, shadow_region

    def fold(self, region: CircuitRegionLike) -> None:
        """
        Fold the specified `region` into a CircuitGate.

        Args:
            region (CircuitRegionLike): The region to fold into a
                CircuitGate.

        Raises:
            ValueError: If `region` is invalid or cannot be straightened.
        """
        if len(region) == 0:
            return

        region = self.straighten(region)[0]
        circuit = self.batch_pop(region.points)

        # Remove placeholders if being called from batch_fold
        placeholder_gate_points = [
            (cycle, op.location[0])
            for cycle, op in circuit.operations_with_cycles()
            if isinstance(op.gate, TaggedGate)
            and op.gate.tag == '__fold_placeholder__'
        ]
        if len(placeholder_gate_points) > 0:
            circuit.batch_pop(placeholder_gate_points)

        # Form and insert CircuitGate
        self.insert_gate(
            region.min_cycle,
            CircuitGate(circuit, True),
            sorted(list(region.keys())),
            list(circuit.params),
        )

    def unfold(self, point: CircuitPointLike) -> None:
        """Unfold the CircuitGate at `point` into the circuit."""
        if not isinstance(self[point].gate, CircuitGate):
            raise ValueError('Expected to unfold a CircuitGate.')

        op = self[point]
        circuit: Circuit = op.gate._circuit  # type: ignore
        circuit.set_params(op.params)
        self.replace_with_circuit(point, circuit)

    def unfold_all(self) -> None:
        """Unfold all CircuitGates in the circuit."""
        while any(
            isinstance(gate, CircuitGate)
            for gate in self.gate_set
        ):
            for cycle, op in self.operations_with_cycles():
                if isinstance(op.gate, CircuitGate):
                    self.unfold((cycle, op.location[0]))
                    break

    def surround(
        self,
        point: CircuitPointLike,
        num_qudits: int,
        bounding_region: CircuitRegionLike | None = None,
        fail_quickly: bool = False,
    ) -> CircuitRegion:
        """
        Retrieve the maximal region in this circuit with `point` included.

        Args:
            point (CircuitPointLike): Find a surrounding region for this
                point. This point will be in the final CircuitRegion.

            num_qudits (int): The number of qudits to include in the region.

            bounding_region (CircuitRegionLike | None): An optional
                region that bounds the resulting region.

            fail_quickly (bool): If set to true, will not branch on
                an invalid region. This will lead to a much faster
                result in some cases at the cost of only approximating
                the maximal region.

        Raises:
            IndexError: If `point` is not a valid index.

            ValueError: If `num_qudits` is nonpositive.

            ValueError: If the operation at `point` is too large for
                `num_qudits`.

            ValueError: If `bounding_region` is invalid.

        Notes:
            This algorithm explores outward horizontally as much as possible.
            When a gate is encountered that involves another qudit not
            currently in the region, a decision needs to be made on whether
            that gate will be included or not. These decisions form a tree;
            an exhaustive search is employed to find the maximal region
            from this decision tree.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected an integer num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError(
                f'Expected a positive integer num_qudits, got {num_qudits}.',
            )

        if bounding_region is not None:
            bounding_region = CircuitRegion(bounding_region)

        point = self.normalize_point(point)

        init_op: Operation = self[point]  # Allow starting at an idle point

        if init_op.num_qudits > num_qudits:
            raise ValueError('Gate at point is too large for num_qudits.')

        HalfWire = Tuple[CircuitPoint, str]
        """
        A HalfWire is a point in the circuit and a direction. This
        represents a point to start exploring from and a direction to
        explore in.
        """

        Node = Tuple[
            List[HalfWire],
            Set[Tuple[int, Operation]],
            CircuitLocation,
            Set[CircuitPoint],
        ]
        """
        A Node in the search tree.

        Each node represents a region that may grow further.
        The data structure tracks all HalfWires in the region and
        the set of operations inside the region. During node exploration
        each HalfWire is walked until we find a multi-qudit gate. Multi-
        qudit gates form branches in the tree on whether on the gate
        should be included. The node structure additionally stores the
        set of qudit indices involved in the region currently. Also, we
        track points that have already been explored to reduce repetition.
        """

        # Initialize the frontier
        init_node = (
            [
                (CircuitPoint(point[0], qudit_index), 'left')
                for qudit_index in init_op.location
            ]
            + [
                (CircuitPoint(point[0], qudit_index), 'right')
                for qudit_index in init_op.location
            ],
            {(point[0], init_op)},
            init_op.location,
            {CircuitPoint(point[0], q) for q in init_op.location},
        )

        frontier: list[Node] = [init_node]

        # Track best so far
        def score(node: Node) -> int:
            return sum(op[1].num_qudits for op in node[1])

        best_score = score(init_node)
        best_region = self.get_region({(point[0], init_op.location[0])})

        # Exhaustive Search
        while len(frontier) > 0:
            node = frontier.pop(0)
            _logger.debug('popped node:')
            _logger.debug(node[0])
            _logger.debug(f'Items remaining in the frontier: {len(frontier)}')

            # Evaluate node
            if score(node) > best_score:
                # Calculate region from best node and return
                points = {(cycle, op.location[0]) for cycle, op in node[1]}

                try:
                    best_region = self.get_region(points)
                    best_score = score(node)
                    _logger.debug(f'new best: {best_region}.')

                # Need to reject bad regions
                except ValueError:
                    if fail_quickly:
                        continue

            # Expand node
            absorbed_gates: set[tuple[int, Operation]] = set()
            branches: set[tuple[int, int, Operation]] = set()
            before_branch_half_wires: dict[int, HalfWire] = {}
            for i, half_wire in enumerate(node[0]):

                cycle_index, qudit_index = half_wire[0]
                step = -1 if half_wire[1] == 'left' else 1

                while True:

                    # Take a step
                    cycle_index += step

                    # Stop at edges
                    if cycle_index < 0 or cycle_index >= self.num_cycles:
                        break

                    # Stop when outside bounds
                    if bounding_region is not None:
                        if (cycle_index, qudit_index) not in bounding_region:
                            break

                    # Stop when exploring previously explored points
                    point = CircuitPoint(cycle_index, qudit_index)
                    if point in node[3]:
                        break
                    node[3].add(point)

                    # Continue until next operation
                    if self.is_point_idle(point):
                        continue
                    op: Operation = self[cycle_index, qudit_index]

                    # Gates already in region stop the half_wire
                    if (cycle_index, op) in node[1]:
                        break

                    # Gates already accounted for stop the half_wire
                    if (cycle_index, op) in absorbed_gates:
                        break

                    if (cycle_index, op) in [(c, o) for h, c, o in branches]:
                        break

                    # Absorb single-qudit gates
                    if len(op.location) == 1:
                        absorbed_gates.add((cycle_index, op))
                        continue

                    # Operations that are too large stop the half_wire
                    if len(op.location.union(node[2])) > num_qudits:
                        break

                    # Otherwise branch on the operation
                    branches.add((i, cycle_index, op))

                    # Track state of half wire right before branch
                    prev_point = CircuitPoint(cycle_index - step, qudit_index)
                    before_branch_half_wires[i] = (prev_point, half_wire[1])
                    break

            # Compute children and extend frontier
            for half_wire_index, cycle_index, op in branches:

                child_half_wires = [
                    half_wire
                    for i, half_wire in before_branch_half_wires.items()
                    if half_wire_index != i
                ]

                qudit = node[0][half_wire_index][0].qudit
                direction = node[0][half_wire_index][1]
                left_expansion = [
                    (CircuitPoint(cycle_index, qudit_index), 'left')
                    for qudit_index in op.location
                    if qudit != qudit_index or direction == 'left'
                ]
                right_expansion = [
                    (CircuitPoint(cycle_index, qudit_index), 'right')
                    for qudit_index in op.location
                    if qudit != qudit_index or direction == 'right'
                ]
                expansion = left_expansion + right_expansion

                # Branch/Gate not taken
                frontier.append((
                    child_half_wires,
                    node[1] | absorbed_gates,
                    node[2],
                    node[3],
                ))

                # Branch/Gate taken
                op_points = {CircuitPoint(cycle_index, q) for q in op.location}
                frontier.append((
                    list(set(child_half_wires + expansion)),
                    node[1] | absorbed_gates | {(cycle_index, op)},
                    node[2].union(op.location),
                    node[3] | op_points,
                ))

            # Append terminal node to handle absorbed gates with no branches
            if len(node[1] | absorbed_gates) != len(node[1]):
                frontier.append(([], node[1] | absorbed_gates, *node[2:]))

        return best_region

    def get_region(self, points: Iterable[CircuitPointLike]) -> CircuitRegion:
        """
        Calculate the minimal region from a sequence of points.

        Args:
            points (Iterable[CircuitPointLike]): The positions of operations
                to group together.

        Returns:
            (CircuitRegion): The region given by `points`.

        Raises:
            ValueError: If `points` do not form a valid convex region
                in the circuit. This happens when there exists an operation
                within the bounds of `points`, but not contained in it.

            IndexError: If any of `points` are out-of-range.
        """

        if not is_iterable(points):
            raise TypeError(
                'Expected iterable of points, got %s.' % type(points),
            )

        points = list(points)

        if not all(CircuitPoint.is_point(point) for point in points):
            checks = [CircuitPoint.is_point(point) for point in points]
            raise TypeError(
                'Expected iterable of points'
                f', got {points[checks.index(False)]} for at least one point.',
            )

        if len(points) == 0:
            return CircuitRegion({})

        # Collect operations
        ops_and_cycles = list({
            (point[0], self[point])
            for point in points
            if not self.is_point_idle(point)
        })

        # Calculate the bounding region to be folded
        # The region is represented by the max and min cycle for each qudit.
        region: dict[int, tuple[int, int]] = {}

        for point in points:
            # Flip negative indices
            cycle_index = point[0]
            if cycle_index < 0:
                cycle_index = self.num_cycles + cycle_index

            if self.is_point_idle(point):
                continue

            for qudit_index in self[point].location:
                if qudit_index not in region:
                    region[qudit_index] = (self.num_cycles, -1)

                if cycle_index < region[qudit_index][0]:
                    region[qudit_index] = (cycle_index, region[qudit_index][1])

                if cycle_index > region[qudit_index][1]:
                    region[qudit_index] = (region[qudit_index][0], cycle_index)

        # Count operations within region
        ops_in_region = set()
        for qudit_index, bounds in region.items():
            for i, cycle in enumerate(self._circuit[bounds[0]:bounds[1] + 1]):
                op = cycle[qudit_index]
                if op is not None:
                    ops_in_region.add((bounds[0] + i, op))

        # All operations in the region must be getting folded
        if len(ops_and_cycles) != len(ops_in_region):
            raise ValueError(
                'Operations cannot be grouped in a region due to'
                ' another operation in the middle.',
            )

        self.check_region(region)

        return CircuitRegion(region)

    def downsize_region(self, region: CircuitRegionLike) -> CircuitRegion:
        """Remove all idle qudits-cycles in `region` while keeping it valid."""
        return self.get_region(CircuitRegion(region).points)

    def get_operations(
            self,
            points: Iterable[CircuitPointLike],
    ) -> list[Operation]:
        """Retrieve operations from `points` without throwing IndexError."""
        if not is_iterable(points):
            raise TypeError(
                f'Expected a circuit point iterable, got {type(points)}.',
            )

        if not all(CircuitPoint.is_point(point) for point in points):
            raise TypeError(
                f'Expected a circuit point iterable, got {type(points)}.',
            )

        ops: set[tuple[int, Operation]] = set()
        for point in points:
            try:
                ops.add((point[0], self.get_operation(point)))
            except IndexError:
                continue

        return [op_and_cycle[1] for op_and_cycle in ops]

    def get_slice(self, points: Sequence[CircuitPointLike]) -> Circuit:
        """Return a copy of a slice of this circuit."""
        if not all(self.is_point_in_range(point) for point in points):
            raise IndexError('Out-of-range point.')

        # Sort points
        points = sorted(points)

        # Collect operations avoiding duplicates
        points_to_skip: list[CircuitPointLike] = []
        ops_and_cycles: list[tuple[Operation, int]] = []
        for point in points:
            if point in points_to_skip:
                continue
            if self.is_point_idle(point):
                continue

            op = self[point]
            ops_and_cycles.append((op, point[0]))
            points_to_skip.extend((point[0], q) for q in op.location)

        ops = [op for op, _ in ops_and_cycles]
        points = [(cycle, op.location[0]) for op, cycle in ops_and_cycles]

        if len(points) == 0:
            raise IndexError('No operations exists at any of the points.')

        # Form new circuit and return
        qudits = list(set(sum((tuple(op.location) for op in ops), ())))
        qudits = sorted(qudits)
        radixes = [self.radixes[q] for q in qudits]
        circuit = Circuit(len(radixes), radixes)
        for op in ops:
            location = [qudits.index(q) for q in op.location]
            circuit.append(Operation(op.gate, location, op.params))
        return circuit

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

    def set_params(self, params: RealVector) -> None:
        """Set all parameters at once."""
        self.check_parameters(params)
        param_index = 0
        for op in self:
            op.params = list(
                params[param_index: param_index + op.num_params],
            )
            param_index += op.num_params

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
            >>> circ.num_params
            6
            >>> circ.get_param_location(4)
            (1, 0, 1)
        """
        if param_index < 0:
            raise IndexError('Negative parameter index is not supported.')

        count = 0
        for cycle, op in self.operations_with_cycles():
            count += len(op.params)
            if count > param_index:
                param = param_index - (count - len(op.params))
                return (cycle, op.location[0], param)

        raise IndexError('Out-of-range parameter index.')

    # endregion

    # region Circuit Logic Methods

    def get_inverse(self) -> Circuit:
        """Return the circuit's inverse circuit."""
        circuit = Circuit(self.num_qudits, self.radixes)
        for op in reversed(self):
            circuit.append(
                Operation(
                    DaggerGate(op.gate),
                    op.location,
                    op.params,
                ),
            )
        return circuit

    def compress(self) -> None:
        """Compress the circuit's cycles."""
        # TODO
        raise NotImplementedError('Compress is not currently implemented.')

    def renumber_qudits(self, qudit_permutation: Sequence[int]) -> None:
        """
        Permute the qudits in the circuit.

        Args:
            qudit_permutation (Sequence[int]): A map from qudit indices
                to qudit indices.

        Raises:
            IndexError: If any of the indices are out of range.

            ValueError: If the `qudit_permutation` is not the same size
                as the circuit.

            ValueError: If the `qudit_permutation` is not a valid permutation.
        """
        if not is_sequence_of_int(qudit_permutation):
            raise TypeError(
                'Expected sequence of integers'
                ', got %s' % type(qudit_permutation),
            )

        if len(qudit_permutation) != self.num_qudits:
            raise ValueError(
                'Expected qudit_permutation length equal to circuit num_qudits:'
                '%d, got %d' % (self.num_qudits, len(qudit_permutation)),
            )

        if len(qudit_permutation) != len(set(qudit_permutation)):
            raise ValueError('Invalid permutation.')

        perm = [int(q) for q in qudit_permutation]

        for i in range(self.num_cycles):
            self._circuit[i] = [
                self._circuit[i][perm.index(q)]
                for q in range(self.num_qudits)
            ]

            qudits_to_skip: list[int] = []
            for j in range(self.num_qudits):
                if j in qudits_to_skip:
                    continue

                op = self._circuit[i][j]
                if op is not None:
                    loc = CircuitLocation([perm[q] for q in op.location])
                    op._location = loc
                    qudits_to_skip.extend(loc)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Return the unitary matrix of the circuit.

        Args:
            params (RealVector): Optionally specify parameters
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
        if len(params) != 0:
            self.check_parameters(params)
            param_index = 0

        utry = UnitaryBuilder(self.num_qudits, self.radixes)

        for op in self:
            if len(params) != 0:
                gparams = params[param_index:param_index + op.num_params]
                utry.apply_right(op.get_unitary(gparams), op.location)
                param_index += op.num_params
            else:
                utry.apply_right(op.get_unitary(), op.location)

        return utry.get_unitary()

    def get_statevector(self, in_state: StateLike) -> StateVector:
        """Calculate the output state given the `in_state` input state."""
        # TODO: Can be made a lot more efficient.
        return self.get_unitary().get_statevector(in_state)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """Return the gradient of the circuit."""
        return self.get_unitary_and_grad(params)[1]

    def get_unitary_and_grad(
        self, params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """Return the unitary and gradient of the circuit."""
        if len(params) != 0:
            self.check_parameters(params)
            param_index = 0

        # Collect matrices, gradients, and locations
        matrices = []
        grads = []
        locations = []

        for op in self:
            if len(params) != 0:
                gparams = params[param_index:param_index + op.num_params]
                param_index += op.num_params
                M, dM = op.get_unitary_and_grad(gparams)
                matrices.append(M)
                grads.append(dM)
                locations.append(op.location)
            else:
                M, dM = op.get_unitary_and_grad()
                matrices.append(M)
                grads.append(dM)
                locations.append(op.location)

        # Calculate gradient
        left = UnitaryBuilder(self.num_qudits, self.radixes)
        right = UnitaryBuilder(self.num_qudits, self.radixes)
        full_gards = []

        for M, loc in zip(matrices, locations):
            right.apply_right(M, loc)

        for M, dM, loc in zip(matrices, grads, locations):
            perm = PermutationMatrix.from_qubit_location(self.num_qudits, loc)
            permT = perm.T
            iden = np.identity(2 ** (self.num_qudits - len(loc)))

            right.apply_left(M, loc, inverse=True)
            right_utry = right.get_unitary()
            left_utry = left.get_unitary()
            for grad in dM:
                # TODO: use tensor contractions here instead of mm
                # Should work fine with non unitary gradients
                # TODO: Fix for non qubits
                full_grad = np.kron(grad, iden)
                full_grad = perm @ full_grad @ permT
                full_gards.append(right_utry @ full_grad @ left_utry)
            left.apply_right(M, loc)

        return left.get_unitary(), np.array(full_gards)

    def instantiate(
        self,
        target: StateLike | UnitaryLike,
        method: str | None = None,
        multistarts: int = 1,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Circuit:
        """
        Instantiate the circuit with respect to a target state or unitary.

        Attempts to change the parameters of the circuit such that the
        circuit either implements the target unitary or maps the zero
        state to the target state.

        Args:
            target (StateLike | UnitaryLike): The target unitary or state.
                If a unitary is specified, the method changes the circuit's
                parameters in an effort to get closer to implementing the
                target. If a state is specified, the method changes the
                circuit's parameters in an effort to get closer to producing
                the target state when starting from the zero state.

            method (str | None): The method with which to instantiate
                the circuit. Currently, `"qfactor"` and `"minimization"`
                are supported. If left None, attempts to pick best method.

            multistarts (int): The number of instantiation jobs to spawn
                and manage. (Default: 1)

            seed (int | None): The seed for any pseudo-random number generators
                to use. Note that this is not guaranteed to make this method
                reproducible.

            kwargs (dict[str, Any]): Method specific options, passed
                directly to method constructor. For more info, see
                `bqskit.ir.opt.instantiaters`.

        Returns:
            Circuit: A reference to `self` is returned

        Raises:
            ValueError: If `method` is invalid.

            ValueError: If `circuit` is incompatible with any method.

            ValueError: If `target` dimension doesn't match with circuit.

            ValueError: If `multistarts` is not a positive integer.

            ValueError: If `seed` is not an integer or `None`
        """
        # Set seed if specified
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(
                    f'Expected seed to be an integer got {type(seed)}.',
                )
            seed_random_sources(seed)

        # Construct instantiater from method
        instantiater: Instantiater | None = None

        err = ''
        for inst in instantiater_order:
            # If method is specified; match it
            if inst.get_method_name().lower() == method:  # type: ignore
                if not inst.is_capable(self):  # type: ignore
                    raise ValueError(
                        'Circuit cannot be instantiated using the '
                        f'{method} method.'
                        f'\n{inst.get_violation_report(self)}',  # type: ignore
                    )
                instantiater = inst(**kwargs)
                break

            # If method is not specified; find first capable one
            if method is None:
                if inst.is_capable(self):  # type: ignore
                    instantiater = inst(**kwargs)
                    break
                err += inst.get_violation_report(self) + '\n'  # type: ignore

        if instantiater is None:
            if err != '':
                raise ValueError(f'No capable instantiater.\n{err}')
            raise ValueError(f'No such instantiatation method {method}.')

        # Check Target
        if is_square_matrix(target):
            target = UnitaryMatrix(target)  # type: ignore
        elif is_vector(target):
            target = StateVector(target)  # type: ignore
        else:
            raise TypeError(
                'Expected either StateVector or UnitaryMatrix'
                ' for target, got %s.' % type(target),
            )

        if target.dim != self.dim:
            raise ValueError('Target dimension mismatch with circuit.')

        # Generate starting points
        starts = instantiater.gen_starting_points(multistarts, self, target)

        # Instantiate the circuit
        params_list = []
        for start in starts:
            params_list.append(instantiater.instantiate(self, target, start))

        # Return best result
        if multistarts == 1:
            self.set_params(params_list[0])
            return self

        cost_fn = HilbertSchmidtCost(self, target)
        self.set_params(sorted(params_list, key=lambda x: cost_fn(x))[0])
        return self

    def minimize(self, cost: CostFunction, **kwargs: Any) -> None:
        """
        Minimize the circuit's cost with respect to some CostFunction.

        Attempts to change the parameters of the circuit such that the
        circuit's cost according to `cost` is best minimized.

        Args:
            cost (CostFunction): The cost function to use when evaluting
                the circuit's cost.

            minimizer (str): The minimization method to use. If unspecified,
                attempts to assign best method. (kwarg)
        """
        minimizer = kwargs.get('minimizer', CeresMinimizer())
        self.set_params(minimizer.minimize(cost, self.params))

    # endregion

    # region Collection and Iteration Methods

    @overload
    def __getitem__(self, point: CircuitPointLike) -> Operation:
        ...

    @overload
    def __getitem__(
            self, points: Iterable[CircuitPointLike],
    ) -> list[Operation]:
        ...

    @overload
    def __getitem__(self, region: CircuitRegionLike) -> list[Operation]:
        ...

    @overload
    def __getitem__(
        self,
        slice: int | slice,
    ) -> list[Operation]:
        ...

    @overload
    def __getitem__(
        self,
        slices: tuple[Iterable[int] | slice, Iterable[int] | slice]
            | tuple[int, Iterable[int] | slice]
            | tuple[Iterable[int] | slice, int],
    ) -> list[Operation]:
        ...

    def __getitem__(
        self,
        indices: CircuitPointLike
            | Iterable[CircuitPointLike]
            | CircuitRegionLike
            | int | Iterable[int] | slice
            | tuple[Iterable[int] | slice, Iterable[int] | slice]
            | tuple[int, Iterable[int] | slice]
            | tuple[Iterable[int] | slice, int],
    ) -> Operation | list[Operation]:
        """
        Retrieve operations from the circuit.

        Args:
            indices (
                CircuitPointLike
                | Iterable[CircuitPointLike]
                | CircuitRegionLike
                | int | slice
                | tuple[Iterable[int] | slice, Iterable[int] | slice]
                | tuple[int, Iterable[int] | slice]
                | tuple[Iterable[int] | slice, int],
            ):
                This parameter describes the area in the circuit to retrieve
                operations from. If a point is given, then the operation
                at that point will be returned. In all other cases,
                a list of operations will be returned. You can specify
                a sequence of points to directly get operations from all
                those points. You can specify a cycle index, a sequence of
                cycle indices, or a slice to retrieve operations from those
                cycles. You can also specify a tuple that describes all
                cycles and qudits to sample from. Lastly, you can specify
                a circuit region to get all the operations there.

        Returns:
            (Operation | list[Operation]): Either a specific operation is
                returned or a list of them depending on the type of `indices`.

        Raises:
            IndexError: If any specified point is invalid or out-of-range.
        """

        if CircuitPoint.is_point(indices):
            return self.get_operation(indices)

        if is_iterable(indices):
            if all(CircuitPoint.is_point(point) for point in indices):
                return self.get_operations(indices)

        if CircuitRegion.is_region(indices):
            return self[CircuitRegion(indices).points]

        if is_integer(indices):
            return list({
                op
                for op in self._circuit[indices]
                if op is not None
            })

        # if is_iterable(indices):
        #     if all(is_integer(cycle_index) for cycle_index in indices):
        #         return sum([self[cycle_index] for cycle_index in indices], [])

        if isinstance(indices, slice):
            start, stop, step = indices.indices(self.num_cycles)
            return sum((self[index] for index in range(start, stop, step)), [])

        if isinstance(indices, tuple) and len(indices) == 2:
            cycle_indices, qudit_indices = indices
            cycles, qudits = None, None

            if is_integer(cycle_indices):
                cycles = [cycle_indices]

            elif isinstance(cycle_indices, slice):
                start, stop, step = cycle_indices.indices(self.num_cycles)
                cycles = list(range(start, stop, step))

            elif is_iterable(cycle_indices):
                if all(is_integer(index) for index in cycle_indices):
                    cycles = list(cycle_indices)

            if is_integer(qudit_indices):
                qudits = [qudit_indices]

            elif isinstance(qudit_indices, slice):
                start, stop, step = qudit_indices.indices(self.num_qudits)
                qudits = list(range(start, stop, step))

            elif is_iterable(qudit_indices):
                if all(is_integer(index) for index in qudit_indices):
                    qudits = list(qudit_indices)

            if cycles is not None and qudits is not None:
                return self[[
                    CircuitPoint(cycle, qudit)
                    for cycle in cycles
                    for qudit in qudits
                ]]

        raise TypeError(
            'Invalid index type. Expected point'
            ', sequence of points, slice, region, or pair of indices'
            ', got %s.' % type(indices),
        )

    def __iter__(self) -> Iterator[Operation]:
        """Return an iterator that iterates through all operations."""
        return CircuitIterator(self)  # type: ignore

    def __reversed__(self) -> Iterator[Operation]:
        """Return a reverse iterator that iterates through all operations."""
        return CircuitIterator(self, reverse=True)  # type: ignore

    def __contains__(self, op: object) -> bool:
        """Return true if `op` is in the circuit."""
        if isinstance(op, Operation):
            try:
                self.check_valid_operation(op)
            except ValueError:
                return False

            if op.gate not in self._gate_info:
                return False

            for _op in self.operations(qudits_or_region=[op.location[0]]):
                if op == _op:
                    return True

            return False

        elif isinstance(op, Gate):
            return op in self._gate_info

        else:
            return False

    def __len__(self) -> int:
        """Return the number of operations in the circuit."""
        return self.num_operations

    def operations(
        self,
        start: CircuitPointLike = CircuitPoint(0, 0),
        end: CircuitPointLike | None = None,
        qudits_or_region: CircuitRegionLike | Sequence[int] | None = None,
        exclude: bool = False,
        reverse: bool = False,
    ) -> Iterator[Operation]:
        """Create and return an iterator, for more info see CircuitIterator."""
        return CircuitIterator(
            self,
            start,
            end,
            qudits_or_region,
            exclude,
            reverse,
        )  # type: ignore

    def operations_with_cycles(
        self,
        start: CircuitPointLike = CircuitPoint(0, 0),
        end: CircuitPointLike | None = None,
        qudits_or_region: CircuitRegionLike | Sequence[int] | None = None,
        exclude: bool = False,
        reverse: bool = False,
    ) -> Iterator[tuple[int, Operation]]:
        """Create and return an iterator, for more info see CircuitIterator."""
        return CircuitIterator(
            self,
            start,
            end,
            qudits_or_region,
            exclude,
            reverse,
            True,
        )  # type: ignore

    # endregion

    # region Operator Overloads

    def __invert__(self) -> Circuit:
        """Invert the circuit."""
        return self.get_inverse()

    def __eq__(self, rhs: object) -> bool:
        """
        Check for circuit equality.

        Two circuits are equal if: 1) They have the same number of operations.
        2) All qudit radixes are equal. 3) All operations in simulation order
        are equal.
        """
        if not isinstance(rhs, Circuit):
            raise NotImplemented

        if self is rhs:
            return True

        if self._gate_info != rhs._gate_info:
            return False

        for r1, r2 in zip(self.radixes, rhs.radixes):
            if r1 != r2:
                return False

        return all(op1 == op2 for op1, op2 in zip(self, rhs))

    def __ne__(self, rhs: object) -> bool:
        """Check for circuit inequality, see __eq__ for more info."""
        return not self == rhs

    def __add__(self, rhs: Circuit) -> Circuit:
        """Return a concatenated circuit copy."""
        circuit = Circuit(self.num_qudits, self.radixes)
        circuit.append_circuit(self, list(range(self.num_qudits)))
        circuit.append_circuit(rhs, list(range(self.num_qudits)))
        return circuit

    def __mul__(self, rhs: int) -> Circuit:
        """Return a repeated circuit copy."""
        circuit = Circuit(self.num_qudits, self.radixes)
        for x in range(rhs):
            circuit.append_circuit(self, list(range(self.num_qudits)))
        return circuit

    def __radd__(self, lhs: Circuit) -> Circuit:
        """Return a concatenated circuit copy."""
        circuit = Circuit(self.num_qudits, self.radixes)
        circuit.append_circuit(lhs, list(range(self.num_qudits)))
        circuit.append_circuit(self, list(range(self.num_qudits)))
        return circuit

    def __iadd__(self, rhs: Circuit) -> None:
        """Return a concatenated circuit copy."""
        self.append_circuit(rhs, list(range(self.num_qudits)))

    def __imul__(self, rhs: int) -> None:
        """Return a repeated circuit copy."""
        circuit = self.copy()
        for x in range(rhs - 1):
            self.append_circuit(circuit, list(range(self.num_qudits)))

    # endregion

    # region IO Methods

    def __str__(self) -> str:
        """String representation of the circuit."""
        op_string = '['
        if self.num_operations == 1:
            op_string += str(list(self.operations())[0])
        elif self.num_operations == 2:
            op_string += str(list(self.operations())[0])
            op_string += ', '
            op_string += str(list(self.operations())[1])
        elif self.num_operations > 2:
            op_string += str(list(self.operations())[0])
            op_string += ' ... '
            op_string += str(list(self.operations())[-1])
        op_string += ']'
        num_qudits = self.num_qudits
        return f'Circuit({num_qudits}){op_string}'

    def __repr__(self) -> str:
        """Repr representation of the circuit."""
        string = 'Circuit(self.num_qudits)'
        for cycle in self._circuit[:100]:
            string += f'\n\t{cycle}'
        if self.num_cycles > 100:
            string += '...'
        return string

    def save(self, filename: str) -> None:
        """Save the circuit to a file."""
        language = get_language(filename.split('.')[-1])

        with open(filename, 'w') as f:
            f.write(language.encode(self))

    def to(self, type: str) -> str:
        """Convert circuit to language."""
        return get_language(type).encode(self)

    @staticmethod
    def from_file(filename: str) -> Circuit:
        """Restore a circuit from a file."""
        language = get_language(filename.split('.')[-1])

        with open(filename) as f:
            return language.decode(f.read())

    @staticmethod
    def from_unitary(utry: UnitaryLike) -> Circuit:
        """Construct a circuit from a single unitary."""
        utry = UnitaryMatrix(utry)
        circuit = Circuit(utry.num_qudits, utry.radixes)
        circuit.append_gate(
            ConstantUnitaryGate(utry), list(range(utry.num_qudits)),
        )
        return circuit

    @staticmethod
    def from_operation(op: Operation) -> Circuit:
        """Construct a circuit from a single operation."""
        circuit = Circuit(op.num_qudits, op.radixes)
        circuit.append_gate(op.gate, list(range(circuit.num_qudits)), op.params)
        return circuit

    # endregion

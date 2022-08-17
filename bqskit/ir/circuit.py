"""This module implements the Circuit class."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from distributed import get_client
from distributed import secede

from bqskit.ir.gate import Gate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.iterator import CircuitIterator
from bqskit.ir.lang import get_language
from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.instantiaters import instantiater_order
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.multistartgen import MultiStartGenerator
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion
from bqskit.ir.region import CircuitRegionLike
from bqskit.qis.graph import CouplingGraph
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
from bqskit.utils.typing import is_bool
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
        self._graph_info: dict[tuple[int, int], int] = {}

        _NodePtrs = Dict[int, Optional[CircuitPoint]]
        self._front: _NodePtrs = {i: None for i in range(self.num_qudits)}
        self._rear: _NodePtrs = {i: None for i in range(self.num_qudits)}
        self._dag: dict[CircuitPoint, tuple[_NodePtrs, _NodePtrs]] = {}

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
    def coupling_graph(self) -> CouplingGraph:
        """
        The qudit connectivity in the circuit.

        Returns:
            (CouplingGraph): The coupling graph required by the circuit.

        Notes:
            - Logical multi-qudit gates require participating qudits to
                have all-to-all connectivity.
        """
        return CouplingGraph(self._graph_info.keys(), self.num_qudits)

    @property
    def gate_set(self) -> set[Gate]:
        """The set of gates in the circuit."""
        return set(self._gate_info.keys())

    @property
    def gate_counts(self) -> dict[Gate, int]:
        """The count of each type of gate in the circuit."""
        return {gate: self.count(gate) for gate in self.gate_set}

    @property
    def active_qudits(self) -> list[int]:
        """The qudits involved in at least one operation."""
        active_qudits = set()  # TODO: add test case for single-qudit gates
        for edge in self._graph_info.keys():
            active_qudits.add(edge[0])
            active_qudits.add(edge[1])
        for qudit, point in self._front.items():
            if point is not None:
                active_qudits.add(qudit)
        return list(sorted(active_qudits))

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

        self._rear[self.num_qudits - 1] = None
        self._front[self.num_qudits - 1] = None

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
        shift_index = lambda q: q if q < qudit_index else q + 1
        inc_point = lambda p: CircuitPoint(p.cycle, p.qudit + 1)
        shift_point = lambda p: p if p.qudit < qudit_index else inc_point(p)
        shift_point_or_none = lambda p: None if p is None else shift_point(p)
        shift_edge = lambda e: (shift_index(e[0]), shift_index(e[1]))

        for cycle in self._circuit:
            cycle.insert(qudit_index, None)

            # Renumber gates with now-invalid locations
            qudits_to_skip: list[int] = []
            for i, op in enumerate(cycle[qudit_index:]):
                if op is None or i + qudit_index in qudits_to_skip:
                    continue
                shifted_location = [shift_index(i) for i in op.location]
                op._location = CircuitLocation(shifted_location)
                qudits_to_skip.extend(op.location)

        # Shift _front, _rear, _graph_info, _dag
        self._front = {
            shift_index(q): shift_point_or_none(p)
            for q, p in self._front.items()
        }
        self._front[qudit_index] = None
        self._rear = {
            shift_index(q): shift_point_or_none(p)
            for q, p in self._rear.items()
        }
        self._rear[qudit_index] = None
        self._graph_info = {
            shift_edge(e): i
            for e, i in self._graph_info.items()
        }
        self._dag = {
            shift_point(p): (
                # Prev
                {
                    shift_index(q): shift_point_or_none(p2)
                    for q, p2 in ptr_map[0].items()
                },
                # Next
                {
                    shift_index(q): shift_point_or_none(p2)
                    for q, p2 in ptr_map[1].items()
                },
            )
            for p, ptr_map in self._dag.items()
        }

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
        shift_index = lambda q: q if q < qudit_index else q - 1
        dec_point = lambda p: CircuitPoint(p.cycle, p.qudit - 1)
        shift_point = lambda p: p if p.qudit < qudit_index else dec_point(p)
        shift_point_or_none = lambda p: None if p is None else shift_point(p)
        shift_edge = lambda e: (shift_index(e[0]), shift_index(e[1]))

        for cycle_index, cycle in enumerate(self._circuit):
            cycle.pop(qudit_index)

        # Renumber gates with now-invalid locations
        for cycle_index, cycle in enumerate(self._circuit):
            qudits_to_skip: list[int] = []
            for i, op in enumerate(cycle[qudit_index:]):
                if op is None or i + qudit_index in qudits_to_skip:
                    continue
                shifted_location = [shift_index(i) for i in op.location]
                op._location = CircuitLocation(shifted_location)
                qudits_to_skip.extend(op.location)

        # Shift _front, _rear, _graph_info, _dag
        self._front = {
            shift_index(q): shift_point_or_none(p)
            for q, p in self._front.items()
            if q != qudit_index
        }
        self._rear = {
            shift_index(q): shift_point_or_none(p)
            for q, p in self._rear.items()
            if q != qudit_index
        }
        self._graph_info = {
            shift_edge(e): i
            for e, i in self._graph_info.items()
        }
        self._dag = {
            shift_point(p): (
                # Prev
                {
                    shift_index(q): shift_point_or_none(p2)
                    for q, p2 in ptr_map[0].items()
                },
                # Next
                {
                    shift_index(q): shift_point_or_none(p2)
                    for q, p2 in ptr_map[1].items()
                },
            )
            for p, ptr_map in self._dag.items()
        }

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

        return self._front[qudit_index] is None

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

        perm_point = lambda p: CircuitPoint(p.cycle, perm[p.qudit])
        perm_point_or_none = lambda p: perm_point(p) if p is not None else p

        self._graph_info = {
            (perm[e[0]], perm[e[1]]): i
            for e, i in self._graph_info.items()
        }
        self._front = {
            perm[i]: perm_point_or_none(p)
            for i, p in self._front.items()
        }
        self._rear = {
            perm[i]: perm_point_or_none(p)
            for i, p in self._rear.items()
        }
        self._dag = {
            perm_point(p): (
                # Prev
                {
                    perm[q]: perm_point_or_none(p2)
                    for q, p2 in ptr_map[0].items()
                },
                # Next
                {
                    perm[q]: perm_point_or_none(p2)
                    for q, p2 in ptr_map[1].items()
                },
            )
            for p, ptr_map in self._dag.items()
        }

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

    # endregion

    # region Cycle Methods

    def _append_cycle(self) -> None:
        """Appends an idle cycle to the end of the circuit."""
        self._circuit.append([None] * self.num_qudits)

    def _insert_cycle(self, cycle_index: int) -> None:
        """Inserts an idle cycle in the circuit."""
        self._circuit.insert(cycle_index, [None] * self.num_qudits)
        inc_point = lambda p: CircuitPoint(p.cycle + 1, p.qudit)
        shift_point = lambda p: p if p.cycle < cycle_index else inc_point(p)
        shift_point_or_none = lambda p: None if p is None else shift_point(p)
        self._front = {
            q: shift_point_or_none(p)
            for q, p in self._front.items()
        }
        self._rear = {
            q: shift_point_or_none(p)
            for q, p in self._rear.items()
        }
        self._dag = {
            shift_point(p): (
                # Prev
                {
                    q: shift_point_or_none(p2)
                    for q, p2 in prevs.items()
                },
                # Next
                {
                    q: shift_point_or_none(p2)
                    for q, p2 in nexts.items()
                },
            )
            for p, (prevs, nexts) in self._dag.items()
        }

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

        ops = {op for op in self._circuit[cycle_index] if op is not None}
        points = [(cycle_index, op.location[0]) for op in ops]
        if len(points) != 0:
            for point in points:
                self.pop(point)
            return
        else:
            self._circuit.pop(cycle_index)

        dec_point = lambda p: CircuitPoint(p.cycle - 1, p.qudit)
        shift_point = lambda p: p if p.cycle < cycle_index else dec_point(p)
        shift_point_or_none = lambda p: None if p is None else shift_point(p)
        self._front = {
            q: shift_point_or_none(p)
            for q, p in self._front.items()
        }
        self._rear = {
            q: shift_point_or_none(p)
            for q, p in self._rear.items()
        }
        self._dag = {
            shift_point(p): (
                # Prev
                {
                    q: shift_point_or_none(p2)
                    for q, p2 in ptr_map[0].items()
                },
                # Next
                {
                    q: shift_point_or_none(p2)
                    for q, p2 in ptr_map[1].items()
                },
            )
            for p, ptr_map in self._dag.items()
        }

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
            return 0

        cycle = 0
        for q in location:
            rear = self._rear[q]
            if rear is not None:
                cycle = max(cycle, rear.cycle + 1)
        return cycle

    def _find_available_or_append_cycle(
        self,
        location: CircuitLocationLike,
    ) -> int:
        """Find the first available cycle, if none exists append one."""

        available_cycle = self.find_available_cycle(location)

        # If no available cycle
        if available_cycle == self.num_cycles:
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

    # region DAG Methods

    @property
    def front(self) -> set[CircuitPoint]:
        """The positions of operations with no dependencies."""
        front_set = set()
        for point in self._front.values():
            if point is not None and len(self.prev(point)) == 0:
                front_set.add(point)
        return front_set

    @property
    def rear(self) -> set[CircuitPoint]:
        """The positions of operations with nothing after them."""
        rear_set = set()
        for point in self._rear.values():
            if point is not None and len(self.next(point)) == 0:
                rear_set.add(point)
        return rear_set

    def next(self, point: CircuitPoint) -> set[CircuitPoint]:
        """Return the points of operations dependent on the one at `point`."""
        return {p for p in self._dag[point][1].values() if p is not None}

    def prev(self, point: CircuitPoint) -> set[CircuitPoint]:
        """Return the points of operations the one at `point` depends on."""
        return {p for p in self._dag[point][0].values() if p is not None}

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
            >>> op = Operation(HGate(), [0])
            >>> circ.append(op) # Appends a Hadamard gate to qudit 0.
        """
        self.check_valid_operation(op)
        cycle_index = self._find_available_or_append_cycle(op.location)
        point = CircuitPoint(cycle_index, op.location[0])

        prevs: dict[int, CircuitPoint | None] = {i: None for i in op.location}
        for qudit_index in op.location:
            # Add op to the circuit structure
            self._circuit[cycle_index][qudit_index] = op

            # Update necessary gates already in _dag to point to this one
            rear = self._rear[qudit_index]
            if rear is not None:
                prevs[qudit_index] = rear
                self._dag[rear][1][qudit_index] = point

            # Update rear pointers
            self._rear[qudit_index] = point

            # Update front pointers
            if self._front[qudit_index] is None:
                self._front[qudit_index] = point

        # Add new op to the dag
        self._dag[point] = (prevs, {i: None for i in op.location})

        # Update _graph_info
        for pair in op.location.pairs:
            if pair not in self._graph_info:
                self._graph_info[pair] = 0
            self._graph_info[pair] += 1

        # Update _gate_info
        if op.gate not in self._gate_info:
            self._gate_info[op.gate] = 0
        self._gate_info[op.gate] += 1

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
        self.append(Operation(gate, location, params))

    def append_circuit(
        self,
        circuit: Circuit,
        location: CircuitLocationLike,
        as_circuit_gate: bool = False,
    ) -> None:
        """
        Append `circuit` at the qudit location specified.

        Args:
            circuit (Circuit): The circuit to append.

            location (CircuitLocationLike): Apply the circuit to these qudits.

            as_circuit_gate (bool): If true, append `circuit` as a unit
                block (CircuitGate) rather than each operation in `circuit`
                individually. (Default: False)

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

        if not is_bool(as_circuit_gate):
            raise TypeError(f'Expected bool, got: {type(as_circuit_gate)}.')

        if circuit.num_qudits != len(location):
            raise ValueError('Circuit and location size mismatch.')

        if as_circuit_gate:
            op = Operation(CircuitGate(circuit), location, circuit.params)
            self.append(op)
            return

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
            self.append(op)
            return

        if not self.is_cycle_in_range(cycle_index):
            if cycle_index < -self.num_cycles:
                cycle_index = 0
            else:
                self.append(op)
                return

        if cycle_index < 0:
            cycle_index = self.num_cycles + cycle_index

        if not self.is_cycle_unoccupied(cycle_index, op.location):
            self._insert_cycle(cycle_index)

        point = CircuitPoint(cycle_index, op.location[0])

        prevs: dict[int, CircuitPoint | None] = {i: None for i in op.location}
        nexts: dict[int, CircuitPoint | None] = {i: None for i in op.location}
        for qudit_index in op.location:

            # Update front pointers if necessary
            if self._front[qudit_index] is None:
                self._front[qudit_index] = point

            # Update rear pointers if necessary
            if self._rear[qudit_index] is None:
                self._rear[qudit_index] = point

            # Search to left of cycle_index on qudit_index
            for c in reversed(range(cycle_index)):
                prev_op = self._circuit[c][qudit_index]

                # Find first op on qudit_index
                if prev_op is not None:
                    prev_point = CircuitPoint(c, prev_op.location[0])

                    # Update its next to be this
                    self._dag[prev_point][1][qudit_index] = point

                    # Add it to this prev
                    prevs[qudit_index] = prev_point

                    # Stop searching to left
                    break

            # Search to right of cycle_index on qudit_index
            for c in range(cycle_index, self.num_cycles):
                next_op = self._circuit[c][qudit_index]

                # Find first op on qudit_index
                if next_op is not None:
                    next_point = CircuitPoint(c, next_op.location[0])

                    # Update its prev to be this
                    self._dag[next_point][0][qudit_index] = point

                    # Add it to this next
                    nexts[qudit_index] = next_point

                    # Stop searching to right
                    break

        # Update _front if this is new front
        for qudit_index, _prev_point in prevs.items():
            if _prev_point is None:
                self._front[qudit_index] = point

        # Update _rear if this is new rear
        for qudit_index, _next_point in nexts.items():
            if _next_point is None:
                self._rear[qudit_index] = point

        # Add this to _dag
        self._dag[point] = (prevs, nexts)

        # Add op to the circuit structure
        for qudit_index in op.location:
            self._circuit[cycle_index][qudit_index] = op

        # Update _graph_info
        for pair in op.location.pairs:
            if pair not in self._graph_info:
                self._graph_info[pair] = 0
            self._graph_info[pair] += 1

        # Update _gate_info
        if op.gate not in self._gate_info:
            self._gate_info[op.gate] = 0
        self._gate_info[op.gate] += 1

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
        as_circuit_gate: bool = False,
    ) -> None:
        """
        Insert `circuit` at the cycle and location specified.

        Args:
            cycle_index (int): The cycle to insert the circuit.

            circuit (Circuit): The circuit to insert.

            location (CircuitLocationLike): Apply the circuit to these
                qudits.

            as_circuit_gate (bool): If true, append `circuit` as a unit
                block (CircuitGate) rather than each operation in `circuit`
                individually. (Default: False)

        Raises:
            ValueError: If `circuit` is not the same size as `location`.

        See Also:
            :func:`insert`
        """

        if not is_integer(cycle_index):
            raise TypeError(
                f'Expected integer cycle index, got: {cycle_index}.',
            )

        if not is_bool(as_circuit_gate):
            raise TypeError(f'Expected bool, got: {type(as_circuit_gate)}.')

        if not CircuitLocation.is_location(location):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)

        if circuit.num_qudits != len(location):
            raise ValueError('Circuit and location size mismatch.')

        if as_circuit_gate:
            op = Operation(CircuitGate(circuit), location, circuit.params)
            self.insert(cycle_index, op)
            return

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
            >>> circ.append_gate(HGate(), [0])
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
        point = self.normalize_point(point)
        point = CircuitPoint(point[0], op.location[0])

        # Remove it from _dag
        prevs, nexts = self._dag[point]
        for q in op.location:
            prev = prevs[q]
            if prev is not None:
                self._dag[prev][1][q] = nexts[q]

            next = nexts[q]
            if next is not None:
                self._dag[next][0][q] = prevs[q]

            if self._front[q] == point:
                self._front[q] = nexts[q]

            if self._rear[q] == point:
                self._rear[q] = prevs[q]
        self._dag.pop(point)

        # Remove it from _circuit
        for qudit_index in op.location:
            self._circuit[point[0]][qudit_index] = None

        if self._is_cycle_idle(point.cycle):
            self.pop_cycle(point.cycle)

        # Update gate and graph counts
        self._gate_info[op.gate] -= 1
        if self._gate_info[op.gate] <= 0:
            self._gate_info.pop(op.gate)

        for pair in op.location.pairs:
            self._graph_info[pair] -= 1
            if self._graph_info[pair] <= 0:
                self._graph_info.pop(pair)

        return op

    def batch_pop(self, points: Iterable[CircuitPointLike]) -> Circuit:
        """
        Pop all operatons at `points` at once.

        Args:
            points (Iterable[CircuitPointLike]): Remove operations
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
        points = [self.normalize_point(point) for point in points]
        points = [p for p in points if not self.is_point_idle(p)]
        ops_and_cycles = {(self[point], point[0]) for point in points}
        ops_and_cycles = sorted(list(ops_and_cycles), key=lambda x: x[1])

        if len(points) == 0:
            raise IndexError('Must batch pop at least one point.')

        # Pop from circuit
        for op, cycle in reversed(ops_and_cycles):
            self.pop((cycle, op.location[0]))

        # Form new circuit and return
        ops = [op for op, _ in ops_and_cycles]
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

        old_op = self._circuit[point[0]][point[1]]
        if old_op is not None and set(old_op.location) == set(op.location):
            if old_op.location[0] != op.location[0]:
                old_point = CircuitPoint(point[0], old_op.location[0])
                new_point = CircuitPoint(point[0], op.location[0])
                prevs, nexts = self._dag[old_point]
                for q in old_op.location:
                    if self._front[q] == old_point:
                        self._front[q] = new_point
                    if self._rear[q] == old_point:
                        self._rear[q] = new_point
                    prev = prevs[q]
                    if prev is not None:
                        self._dag[prev][1][q] = new_point
                    next = nexts[q]
                    if next is not None:
                        self._dag[next][0][q] = new_point
                self._dag[new_point] = self._dag[old_point]
                self._dag.pop(old_point)

            self._gate_info[old_op.gate] -= 1
            if self._gate_info[old_op.gate] <= 0:
                self._gate_info.pop(old_op.gate)

            if op.gate not in self._gate_info:
                self._gate_info[op.gate] = 0
            self._gate_info[op.gate] += 1

            for q in old_op.location:
                self._circuit[point[0]][q] = op

            return

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
        as_circuit_gate: bool = False,
    ) -> None:
        """Replace the operation at 'point' with `circuit`."""
        op = self.pop(point)

        if circuit.num_qudits != op.num_qudits:
            raise ValueError('Cannot replace operation with circuit.')

        if circuit.radixes != tuple(self.radixes[x] for x in op.location):
            raise ValueError('Cannot replace operation with circuit.')

        self.insert_circuit(point[0], circuit, op.location, as_circuit_gate)

    def copy(self) -> Circuit:
        """Return a deep copy of this circuit."""
        circuit = Circuit(self.num_qudits, self.radixes)
        circuit._circuit = copy.deepcopy(self._circuit)
        circuit._gate_info = copy.deepcopy(self._gate_info)
        circuit._graph_info = copy.deepcopy(self._graph_info)
        circuit._front = copy.deepcopy(self._front)
        circuit._rear = copy.deepcopy(self._rear)
        circuit._dag = copy.deepcopy(self._dag)
        return circuit

    def become(self, circuit: Circuit, deepcopy: bool = True) -> None:
        """Become a copy of `circuit`."""
        if deepcopy:
            self._num_qudits = circuit.num_qudits
            self._radixes = circuit.radixes
            self._circuit = copy.deepcopy(circuit._circuit)
            self._gate_info = copy.deepcopy(circuit._gate_info)
            self._graph_info = copy.deepcopy(circuit._graph_info)
            self._front = copy.deepcopy(circuit._front)
            self._rear = copy.deepcopy(circuit._rear)
            self._dag = copy.deepcopy(circuit._dag)
        else:
            self._num_qudits = circuit.num_qudits
            self._radixes = circuit.radixes
            self._circuit = copy.copy(circuit._circuit)
            self._gate_info = copy.copy(circuit._gate_info)
            self._front = copy.copy(circuit._front)
            self._rear = copy.copy(circuit._rear)
            self._dag = copy.copy(circuit._dag)

    def clear(self) -> None:
        """Clear the circuit."""
        self._circuit = []
        self._gate_info = {}
        self._graph_info = {}
        self._front = {i: None for i in range(self.num_qudits)}
        self._rear = {i: None for i in range(self.num_qudits)}
        self._dag = {}

    def compress(self) -> None:
        """Compress the circuit's cycles."""
        compressed_circuit = Circuit(self.num_qudits, self.radixes)
        for op in self:
            compressed_circuit.append(op)
        self.become(compressed_circuit)

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
                        s_point = CircuitPoint(old_cycle_index, op.location[0])
                        d_point = CircuitPoint(new_cycle_index, op.location[0])

                        prevs, nexts = self._dag[s_point]
                        for qudit in op.location:
                            self._circuit[new_cycle_index][qudit] = op
                            self._circuit[old_cycle_index][qudit] = None
                            prev = prevs[qudit]
                            if prev is not None:
                                self._dag[prev][1][qudit] = d_point
                            next = nexts[qudit]
                            if next is not None:
                                self._dag[next][0][qudit] = d_point
                            if self._front[qudit] == s_point:
                                self._front[qudit] = d_point
                            if self._rear[qudit] == s_point:
                                self._rear[qudit] = d_point
                            self._dag.pop(s_point)
                            self._dag[d_point] = (prevs, nexts)
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

    def fold(self, region: CircuitRegionLike) -> CircuitPoint:
        """
        Fold the specified `region` into a CircuitGate.

        Args:
            region (CircuitRegionLike): The region to fold into a
                CircuitGate.

        Returns:
            (CircuitPoint): The resulting CircuitGate's location.

        Raises:
            ValueError: If `region` is invalid or cannot be straightened.
        """
        if len(region) == 0:
            raise ValueError('Empty region cannot be folded.')

        region = self.straighten(region)[0]
        circuit = self.batch_pop(region.points)

        # Insert popped circuit as a CircuitGate
        self.insert_circuit(
            region.min_cycle,
            circuit,
            sorted(list(region.keys())),
            True,
        )

        return CircuitPoint(region.min_cycle, region.min_qudit)

    def unfold(self, point: CircuitPointLike) -> None:
        """Unfold the CircuitGate at `point` into the circuit."""
        if not isinstance(self[point].gate, CircuitGate):
            raise ValueError('Expected to unfold a CircuitGate.')

        op = self[point]
        circuit: Circuit = op.gate._circuit  # type: ignore
        circuit.set_params(op.params)
        self.replace_with_circuit(point, circuit)

    def batch_unfold(self, points: Sequence[CircuitPointLike]) -> None:
        """Unfold the CircuitGates at `points` into the circuit."""
        points = {(point[0], self[point].location[0]) for point in points}
        for point in reversed(sorted(points)):
            self.unfold(point)

    def unfold_all(self) -> None:
        """Unfold all CircuitGates in the circuit."""
        while any(isinstance(gate, CircuitGate) for gate in self.gate_set):
            unfolded_circuit = Circuit(self.num_qudits, self.radixes)
            for op in self:
                if isinstance(op.gate, CircuitGate):
                    circuit = op.gate._circuit
                    circuit.set_params(op.params)
                    unfolded_circuit.append_circuit(circuit, op.location)
                else:
                    unfolded_circuit.append(op)
            self.become(unfolded_circuit)

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
        ops_and_cycles: list[tuple[Operation, int]] = list({
            (self[point], point[0])
            for point in points
            if not self.is_point_idle(point)
        })

        if len(ops_and_cycles) == 0:
            raise IndexError('No operations exists at any of the points.')

        ops_and_cycles.sort(key=lambda x: (x[1], *x[0].location))
        ops = [op for op, _ in ops_and_cycles]

        if len(ops_and_cycles) == 0:
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
                full_grad = permT @ full_grad @ perm
                full_gards.append(right_utry @ full_grad @ left_utry)
            left.apply_right(M, loc)

        return left.get_unitary(), np.array(full_gards)

    def instantiate(
        self,
        target: StateLike | UnitaryLike,
        method: str | Instantiater | None = None,
        multistarts: int = 1,
        seed: int | None = None,
        multistart_gen: MultiStartGenerator = RandomStartGenerator(),
        score_fn_gen: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
        parallel: bool = False,
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

            method (str | Instantiater | None): The method with which to
                instantiate the circuit. Currently, `"qfactor"` and
                `"minimization"` are supported. If left None, attempts to
                pick best method. You can also pass an :class:`Instantiater`
                directly through this.

            multistarts (int): The number of starting points to sample
                instantiation with. If `parallel` is True and this is greater
                than one, will spawn this many Dask tasks. (Default: 1)

            seed (int | None): The seed for any pseudo-random number generators
                to use. Note that this is not guaranteed to make this method
                reproducible.

            multistart_gen (MultiStartGenerator): The generator used to
                generate starting points for instantiation.
                (Default: RandomStartGenerator())

            score_fn_gen (CostFunctionGenerator): The generator used to produce
                a cost function, which will be used to evaluate the best result
                from the different starting points.
                (Default: HilbertSchmidtCostGenerator())

            parallel (bool): If True and `multistarts` is greater than 1,
                this will attempt to connect to a dask cluster and submit
                jobs to be run in parallel. (Default: False)

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
                    f'Expected seed to be an integer, got {type(seed)}.',
                )
            seed_random_sources(seed)

        # Use given Instantiater if one is specified.
        if isinstance(method, Instantiater):
            instantiater = method
            if not instantiater.is_capable(self):
                raise ValueError(
                    'Circuit cannot be instantiated using the '
                    f'{method} method.'
                    f'\n{instantiater.get_violation_report(self)}',
                )

        # Find best Instantiater if none specified
        elif method is None:
            err = ''
            instantiater = None
            for inst in instantiater_order:
                inst_t = cast(Instantiater, inst)
                if inst_t.is_capable(self):
                    instantiater = inst_t(**kwargs)  # type: ignore
                    break
                err += inst_t.get_violation_report(self) + '\n'
            if instantiater is None:
                raise ValueError(f'No capable instantiater.\n{err}')

        # If method is specified by name; match it
        elif isinstance(method, str):
            instantiater = None
            for inst in instantiater_order:
                inst_t = cast(Instantiater, inst)
                if inst_t.get_method_name().lower() == method.lower():
                    if not inst_t.is_capable(self):
                        raise ValueError(
                            'Circuit cannot be instantiated using the '
                            f'{method} method.'
                            f'\n{inst_t.get_violation_report(self)}',
                        )
                    instantiater = inst_t(**kwargs)  # type: ignore
                    break
            if instantiater is None:
                raise ValueError(f'No such instantiatation method {method}.')

        else:
            raise TypeError(
                'Expected a instantiater or name for method,'
                f' got {type(method)}.',
            )

        instantiater = cast(Instantiater, instantiater)

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
        starts = multistart_gen.gen_starting_points(multistarts, self, target)

        if len(starts) != multistarts:
            raise ValueError(
                'Error generating starting points for instantiation.\n'
                f'Expected {multistarts} starts but got {len(starts)}.',
            )

        # Generate cost function
        if not isinstance(score_fn_gen, CostFunctionGenerator):
            raise TypeError(
                'Expected CostFunctionGenerator, got %s.' % type(score_fn_gen),
            )

        cost_fn = score_fn_gen.gen_cost(self, target)

        # Instantiate the circuit
        if parallel and multistarts > 1:
            client = get_client()

            def single_start_instantiate(
                instantiater: Instantiater,
                circuit: Circuit,
                target: UnitaryMatrix,
                start: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                return instantiater.instantiate(circuit, target, start)

            def scoring_fn(
                fn_gen: CostFunctionGenerator,
                circuit: Circuit,
                target: UnitaryMatrix,
                params: npt.NDArray[np.float64],
            ) -> float:
                return fn_gen.gen_cost(circuit, target).get_cost(params)

            param_futures = client.map(
                single_start_instantiate,
                [instantiater] * multistarts,
                [self] * multistarts,
                [target] * multistarts,
                starts,
                pure=False,
            )

            score_futures = client.map(
                scoring_fn,
                [score_fn_gen] * multistarts,
                [self] * multistarts,
                [target] * multistarts,
                param_futures,
                pure=False,
            )

            # We only want to secede on worker threads, so try to recover if
            # Circuit.instantiate is called from the main thread
            try:
                secede()
            except ValueError:
                pass

            scores = client.gather(score_futures)
            best_index = scores.index(min(scores))
            params = param_futures[best_index].result()

        else:
            params_list = [
                instantiater.instantiate(self, target, start)
                for start in starts
            ]
            params = sorted(params_list, key=lambda x: cost_fn(x))[0]

        # Return best result
        self.set_params(params)
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

    # region Measurement Methods

    def remove_all_measurements(self) -> None:
        """Remove all measurement placeholders from the circuit."""
        while any(isinstance(g, MeasurementPlaceholder) for g in self.gate_set):
            for g in self.gate_set:
                if isinstance(g, MeasurementPlaceholder):
                    self.remove(g)

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
            acm: list[Operation] = []
            return sum((self[index] for index in range(start, stop, step)), acm)

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
        string = f'Circuit({self.num_qudits})'
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

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion


def check_no_idle_cycles(circuit: Circuit) -> None:
    for cycle_index in range(circuit.num_cycles):
        assert not circuit._is_cycle_idle(cycle_index)


class TestFold:
    """This tests `circuit.fold`."""

    @pytest.mark.parametrize(
        'points',
        [
            [(0, 0)],
            [(0, 0), (1, 2)],
            [CircuitPoint(0, 0), (1, 2)],
            [(0, 0), CircuitPoint(1, 2)],
            [CircuitPoint(0, 0), CircuitPoint(1, 2)],
        ],
    )
    def test_type_valid(self, points: Sequence[CircuitPointLike]) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.fold(circuit.get_region(points))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    @pytest.mark.parametrize(
        'not_points',
        [
            5,
            [1, 2],
            [1, 'a'],
            'abc',
        ],
    )
    def test_type_invalid(self, not_points: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        with pytest.raises(TypeError):
            circuit.fold(not_points)

    @pytest.mark.parametrize(
        'points',
        [
            [(0, 0)],
            [(0, 0), (1, 2)],
            [CircuitPoint(0, 0), (1, 2)],
            [(0, 0), CircuitPoint(1, 2)],
            [CircuitPoint(0, 0), CircuitPoint(1, 2)],
        ],
    )
    def test_invalid_points(self, points: Sequence[CircuitPointLike]) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        with pytest.raises(IndexError):
            circuit.fold(circuit.get_region(points))

    # def test_empty(self, r6_qudit_circuit: Circuit) -> None:
    #     num_ops = r6_qudit_circuit.num_operations
    #     gate_set = r6_qudit_circuit.gate_set
    #     r6_qudit_circuit.fold(r6_qudit_circuit.get_region([]))
    #     assert num_ops == r6_qudit_circuit.num_operations
    #     assert gate_set == r6_qudit_circuit.gate_set

    @pytest.mark.parametrize(
        'points',
        [
            [(0, 0), (3, 0)],
            [(3, 0), (0, 0)],
            [(0, 0), (2, 0)],
            [(2, 0), (0, 0)],
        ],
    )
    def test_invalid_fold(self, points: Sequence[CircuitPointLike]) -> None:
        circuit = Circuit(4)
        wide_gate = IdentityGate(4)
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        with pytest.raises(ValueError):
            circuit.fold(circuit.get_region(points))

    def test_correctness_1(self) -> None:
        circuit = Circuit(4)
        wide_gate = IdentityGate(4)
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        circuit.append_gate(wide_gate, [0, 1, 2, 3])
        assert circuit.num_operations == 4
        assert circuit.depth == 4
        utry = circuit.get_unitary()

        circuit.fold(circuit.get_region([(0, 0), (1, 0)]))
        assert circuit.num_operations == 3
        assert circuit.depth == 3
        check_no_idle_cycles(circuit)
        for q in range(4):
            assert isinstance(circuit[0, q].gate, CircuitGate)
        for c in range(1, 3, 1):
            for q in range(4):
                assert isinstance(circuit[c, q].gate, IdentityGate)
                assert isinstance(circuit[c, q].gate, IdentityGate)
        test_gate: CircuitGate = circuit[0, 0].gate  # type: ignore
        assert test_gate._circuit.num_operations == 2
        assert test_gate._circuit.num_cycles == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, IdentityGate)
            assert isinstance(test_gate._circuit[1, q].gate, IdentityGate)

        circuit.fold(circuit.get_region([(1, 0), (2, 0)]))
        assert circuit.num_operations == 2
        assert circuit.depth == 2
        check_no_idle_cycles(circuit)
        for c in range(2):
            for q in range(4):
                assert isinstance(circuit[c, q].gate, CircuitGate)
        test_gate: CircuitGate = circuit[0, 0].gate  # type: ignore
        assert test_gate._circuit.num_operations == 2
        assert test_gate._circuit.num_cycles == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, IdentityGate)
            assert isinstance(test_gate._circuit[1, q].gate, IdentityGate)
        test_gate: CircuitGate = circuit[1, 0].gate  # type: ignore
        assert test_gate._circuit.num_operations == 2
        assert test_gate._circuit.num_cycles == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, IdentityGate)
            assert isinstance(test_gate._circuit[1, q].gate, IdentityGate)

        circuit.fold(circuit.get_region([(0, 0), (1, 0)]))
        assert circuit.num_operations == 1
        assert circuit.depth == 1
        check_no_idle_cycles(circuit)
        for q in range(4):
            assert isinstance(circuit[0, q].gate, CircuitGate)
        test_gate: CircuitGate = circuit[0, 0].gate  # type: ignore
        assert test_gate._circuit.num_operations == 2
        assert test_gate._circuit.num_cycles == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, CircuitGate)
            assert isinstance(test_gate._circuit[1, q].gate, CircuitGate)
        inner_gate1: CircuitGate = test_gate._circuit[0, 0].gate  # type: ignore
        inner_gate2: CircuitGate = test_gate._circuit[1, 0].gate  # type: ignore
        assert inner_gate1._circuit.num_operations == 2
        assert inner_gate1._circuit.num_cycles == 2
        for q in range(4):
            assert isinstance(inner_gate1._circuit[0, q].gate, IdentityGate)
            assert isinstance(inner_gate1._circuit[1, q].gate, IdentityGate)
            assert isinstance(inner_gate2._circuit[0, q].gate, IdentityGate)
            assert isinstance(inner_gate2._circuit[1, q].gate, IdentityGate)

        check_no_idle_cycles(circuit)
        assert np.allclose(utry, circuit.get_unitary())

    def test_correctness_2(self) -> None:
        circuit = Circuit(3)
        wide_gate = IdentityGate(3)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(wide_gate, [0, 1, 2])
        utry = circuit.get_unitary()

        circuit.fold(circuit.get_region([(0, 1), (1, 0)]))
        assert circuit.num_operations == 2
        assert circuit.depth == 2
        assert circuit[0, 0].gate is HGate()
        assert isinstance(circuit[1, 0].gate, CircuitGate)
        test_gate: CircuitGate = circuit[1, 0].gate
        assert test_gate._circuit[0, 1].gate is CNOTGate()
        assert test_gate._circuit[0, 2].gate is CNOTGate()
        assert test_gate._circuit[1, 0].gate is wide_gate
        assert test_gate._circuit[1, 1].gate is wide_gate
        assert test_gate._circuit[1, 2].gate is wide_gate
        check_no_idle_cycles(circuit)
        assert np.allclose(utry, circuit.get_unitary())

    def test_correctness_3(self) -> None:
        circuit = Circuit(5)
        wide_gate = IdentityGate(3)
        circuit.append_gate(HGate(), [1])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(wide_gate, [1, 2, 3])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(HGate(), [3])
        circuit.append_gate(XGate(), [0])
        circuit.append_gate(XGate(), [0])
        circuit.append_gate(XGate(), [0])
        circuit.append_gate(XGate(), [4])
        circuit.append_gate(XGate(), [4])
        circuit.append_gate(XGate(), [4])
        utry = circuit.get_unitary()

        circuit.fold(circuit.get_region([(0, 2), (1, 1), (2, 1)]))
        assert circuit.num_operations == 9
        assert circuit.depth == 3
        assert circuit.count(HGate()) == 2
        assert circuit.count(XGate()) == 6
        assert isinstance(circuit[1, 1].gate, CircuitGate)
        test_gate: CircuitGate = circuit[1, 1].gate
        assert test_gate._circuit[0, 1].gate is CNOTGate()
        assert test_gate._circuit[0, 2].gate is CNOTGate()
        assert test_gate._circuit[1, 0].gate is wide_gate
        assert test_gate._circuit[1, 1].gate is wide_gate
        assert test_gate._circuit[1, 2].gate is wide_gate
        check_no_idle_cycles(circuit)
        assert np.allclose(utry, circuit.get_unitary())

    def test_parameters(self) -> None:
        circ = Circuit(2)
        circ.append_gate(CNOTGate(), [1, 0])
        circ.append_gate(U3Gate(), [0], [0, 0, 0.23])
        circ.append_gate(CNOTGate(), [1, 0])

        before_fold = circ.get_unitary()

        circ.fold(circ.get_region([(0, 0), (1, 0), (2, 0)]))

        after_fold = circ.get_unitary()

        assert after_fold == before_fold

    def test_deterministic_gate_ordering(self) -> None:
        def create_circuit(
            coupling_graph: list[tuple[int, int]],
            regions: list[list[int]],
            iterations: int,
        ) -> Circuit:
            circuit = Circuit(4)
            for location in regions * iterations:
                circuit.append_gate(CNOTGate(), location)
                circuit.append_gate(U3Gate(), [location[0]])
                circuit.append_gate(U3Gate(), [location[1]])
            return circuit

        coupling_graph = [(0, 1), (1, 2), (2, 3)]
        circuits = [
            create_circuit(
                coupling_graph, [[0, 1], [2, 3]], 4,
            ) for x in range(10)
        ]

        for (a, b) in coupling_graph:
            region = {a: (0, circuits[0].depth), b: (0, circuits[0].depth)}

            operations = circuits[0][region]
            for circuit in circuits[1:]:
                assert circuit[region] == operations


class TestSurround:
    """This tests circuit.surround."""

    def test_small_circuit_1(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        region = circuit.surround((0, 1), 2)
        assert region == CircuitRegion({0: (0, 2), 1: (0, 2)})

    def test_small_circuit_2(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), (1, 2))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        region = circuit.surround((0, 1), 2)
        assert region == CircuitRegion({0: (0, 1), 1: (0, 3)})

    def test_small_circuit_3(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), (1, 2))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(HGate(), 0)
        region = circuit.surround((0, 1), 3)
        assert region == CircuitRegion({0: (0, 5), 1: (0, 5), 2: (0, 5)})

    def test_through_middle_of_outside(self) -> None:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 1))
        region = circuit.surround((1, 0), 2)
        assert region == CircuitRegion({0: (0, 1), 1: (0, 2)})

    def test_with_fold(self, r6_qudit_circuit: Circuit) -> None:
        cycle = 0
        qudit = 0
        while True:
            cycle = np.random.randint(r6_qudit_circuit.num_cycles)
            qudit = np.random.randint(r6_qudit_circuit.num_qudits)
            if not r6_qudit_circuit.is_point_idle((cycle, qudit)):
                break
        utry = r6_qudit_circuit.get_unitary()
        region = r6_qudit_circuit.surround((cycle, qudit), 4)
        r6_qudit_circuit.fold(region)
        assert r6_qudit_circuit.get_unitary() == utry

    def test_surround_symmetric(self) -> None:
        circuit = Circuit(6)
        # whole wall of even
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [4, 5])

        # one odd gate; problematic point in test
        circuit.append_gate(CNOTGate(), [3, 4])

        # whole wall of even
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [4, 5])

        region = circuit.surround((1, 3), 4)
        assert region.location == CircuitLocation([2, 3, 4, 5])

    def test_surround_filter_hard(self) -> None:
        circuit = Circuit(7)
        # whole wall of even
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [4, 5])

        # one odd gate; problematic point in test
        circuit.append_gate(CNOTGate(), [3, 4])

        # whole wall of even
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [4, 5])

        # more odd gates to really test filter
        circuit.append_gate(CNOTGate(), [5, 6])
        circuit.append_gate(CNOTGate(), [5, 6])
        circuit.append_gate(CNOTGate(), [5, 6])
        circuit.append_gate(CNOTGate(), [5, 6])
        circuit.append_gate(CNOTGate(), [5, 6])

        region = circuit.surround(
            (1, 3), 4, None, None, lambda region: (
                region.min_qudit > 1 and region.max_qudit < 6
            ),
        )
        assert region.location == CircuitLocation([2, 3, 4, 5])

    def test_surround_filter_topology(self) -> None:
        circuit = Circuit(5)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [3, 4])

        def region_filter(region: CircuitRegion) -> bool:
            return circuit.get_slice(region.points).coupling_graph.is_linear()

        region = circuit.surround(
            (4, 1), 4, None, None, lambda region: (
                region_filter(region)
            ),
        )
        assert circuit.is_valid_region(region)
        assert region.location == CircuitLocation([1, 2, 3, 4])


def test_check_region_1() -> None:
    c = Circuit(4)
    c.append_gate(CNOTGate(), [1, 2])
    c.append_gate(CNOTGate(), [0, 1])
    c.append_gate(CNOTGate(), [2, 3])
    c.append_gate(CNOTGate(), [1, 2])
    assert not c.is_valid_region({1: (0, 2), 2: (0, 2), 3: (0, 2)})


def test_check_region_2() -> None:
    c = Circuit(3)
    c.append_gate(CNOTGate(), [0, 1])
    c.append_gate(CNOTGate(), [0, 2])
    c.append_gate(CNOTGate(), [1, 2])
    assert not c.is_valid_region({0: (0, 0), 1: (0, 2), 2: (2, 2)})

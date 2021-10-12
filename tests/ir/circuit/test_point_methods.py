"""This test module tests circuit point methods."""
from __future__ import annotations

import pytest
from hypothesis import given

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.utils.test.strategies import circuit_point_likes
from bqskit.utils.test.strategies import circuits
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import valid_type_test
from bqskit.utils.typing import is_bool


class TestIsPointInRange:

    @valid_type_test(Circuit(1).is_point_in_range)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).is_point_in_range)
    def test_invalid_type(self) -> None:
        pass

    @given(circuit_point_likes())
    def test_return_type(self, point: CircuitPointLike) -> None:
        circuit = Circuit(1)
        assert is_bool(circuit.is_point_in_range(point))

    @pytest.mark.parametrize(
        'point', [
            (-5, -5),
            (-4, -4),
            (-3, -3),
            (-2, -2),
            (-1, -1),
        ],
    )
    def test_true_neg(self, point: CircuitPointLike) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert circuit.is_point_in_range(point)

    @pytest.mark.parametrize(
        'point', [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
        ],
    )
    def test_true_pos(self, point: CircuitPointLike) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert circuit.is_point_in_range(point)

    @pytest.mark.parametrize(
        'point', [
            (-1000, 0),
            (1, -100),
            (-8, -8),
            (-6, -6),
            (-7, 4),
        ],
    )
    def test_false_neg(self, point: CircuitPointLike) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert not circuit.is_point_in_range(point)

    @pytest.mark.parametrize(
        'point', [
            (1000, 0),
            (1, 100),
            (8, 8),
            (6, 6),
            (5, 4),
            (3, 8),
            (2, 9),
            (8, 2),
        ],
    )
    def test_false_pos(self, point: CircuitPointLike) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert not circuit.is_point_in_range(point)


class TestIsPointIdle:

    @valid_type_test(Circuit(1).is_point_idle)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).is_point_idle)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits())
    def test_return_type(self, circuit: Circuit) -> None:
        for cycle in range(circuit.num_cycles):
            for qudit in range(circuit.num_qudits):
                assert is_bool(circuit.is_point_idle((cycle, qudit)))

    @given(circuits())
    def test_not_idle(self, circuit: Circuit) -> None:
        points = set()
        for cycle, op in circuit.operations_with_cycles():
            for qudit in op.location:
                assert not circuit.is_point_idle((cycle, qudit))
                points.add((cycle, qudit))

        for cycle in range(circuit.num_cycles):
            for qudit in range(circuit.num_qudits):
                if (cycle, qudit) not in points:
                    assert circuit.is_point_idle((cycle, qudit))


class TestNormalizePoint:

    @valid_type_test(Circuit(1).normalize_point)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).normalize_point)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits())
    def test_normalize(self, circuit: Circuit) -> None:
        for cycle in range(-circuit.num_cycles, circuit.num_cycles):
            for qudit in range(-circuit.num_qudits, circuit.num_qudits):
                point = (cycle, qudit)
                norm_point = circuit.normalize_point(point)
                assert isinstance(norm_point, CircuitPoint)
                cell1 = circuit._circuit[point[0]][point[1]]
                cell2 = circuit._circuit[norm_point[0]][norm_point[1]]
                assert cell1 == cell2
                assert 0 <= norm_point.qudit < circuit.num_qudits
                assert 0 <= norm_point.cycle < circuit.num_cycles

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
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.ir.region import CircuitRegion


def check_no_idle_cycles(circuit: Circuit) -> None:
    for cycle_index in range(circuit.get_num_cycles()):
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
    #     num_ops = r6_qudit_circuit.get_num_operations()
    #     gate_set = r6_qudit_circuit.get_gate_set()
    #     r6_qudit_circuit.fold(r6_qudit_circuit.get_region([]))
    #     assert num_ops == r6_qudit_circuit.get_num_operations()
    #     assert gate_set == r6_qudit_circuit.get_gate_set()

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
        assert circuit.get_num_operations() == 4
        assert circuit.get_depth() == 4
        utry = circuit.get_unitary()

        circuit.fold(circuit.get_region([(0, 0), (1, 0)]))
        assert circuit.get_num_operations() == 3
        assert circuit.get_depth() == 3
        check_no_idle_cycles(circuit)
        for q in range(4):
            assert isinstance(circuit[0, q].gate, CircuitGate)
        for c in range(1, 3, 1):
            for q in range(4):
                assert isinstance(circuit[c, q].gate, IdentityGate)
                assert isinstance(circuit[c, q].gate, IdentityGate)
        test_gate: CircuitGate = circuit[0, 0].gate  # type: ignore
        assert test_gate._circuit.get_num_operations() == 2
        assert test_gate._circuit.get_num_cycles() == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, IdentityGate)
            assert isinstance(test_gate._circuit[1, q].gate, IdentityGate)

        circuit.fold(circuit.get_region([(1, 0), (2, 0)]))
        assert circuit.get_num_operations() == 2
        assert circuit.get_depth() == 2
        check_no_idle_cycles(circuit)
        for c in range(2):
            for q in range(4):
                assert isinstance(circuit[c, q].gate, CircuitGate)
        test_gate: CircuitGate = circuit[0, 0].gate  # type: ignore
        assert test_gate._circuit.get_num_operations() == 2
        assert test_gate._circuit.get_num_cycles() == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, IdentityGate)
            assert isinstance(test_gate._circuit[1, q].gate, IdentityGate)
        test_gate: CircuitGate = circuit[1, 0].gate  # type: ignore
        assert test_gate._circuit.get_num_operations() == 2
        assert test_gate._circuit.get_num_cycles() == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, IdentityGate)
            assert isinstance(test_gate._circuit[1, q].gate, IdentityGate)

        circuit.fold(circuit.get_region([(0, 0), (1, 0)]))
        assert circuit.get_num_operations() == 1
        assert circuit.get_depth() == 1
        check_no_idle_cycles(circuit)
        for q in range(4):
            assert isinstance(circuit[0, q].gate, CircuitGate)
        test_gate: CircuitGate = circuit[0, 0].gate  # type: ignore
        assert test_gate._circuit.get_num_operations() == 2
        assert test_gate._circuit.get_num_cycles() == 2
        for q in range(4):
            assert isinstance(test_gate._circuit[0, q].gate, CircuitGate)
            assert isinstance(test_gate._circuit[1, q].gate, CircuitGate)
        inner_gate1: CircuitGate = test_gate._circuit[0, 0].gate  # type: ignore
        inner_gate2: CircuitGate = test_gate._circuit[1, 0].gate  # type: ignore
        assert inner_gate1._circuit.get_num_operations() == 2
        assert inner_gate1._circuit.get_num_cycles() == 2
        for q in range(4):
            assert isinstance(inner_gate1._circuit[0, q].gate, IdentityGate)
            assert isinstance(inner_gate1._circuit[1, q].gate, IdentityGate)
            assert isinstance(inner_gate2._circuit[0, q].gate, IdentityGate)
            assert isinstance(inner_gate2._circuit[1, q].gate, IdentityGate)

        check_no_idle_cycles(circuit)
        assert np.allclose(utry.get_numpy(), circuit.get_unitary().get_numpy())

    def test_correctness_2(self) -> None:
        circuit = Circuit(3)
        wide_gate = IdentityGate(3)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(wide_gate, [0, 1, 2])
        utry = circuit.get_unitary()

        circuit.fold(circuit.get_region([(0, 1), (1, 0)]))
        assert circuit.get_num_operations() == 2
        assert circuit.get_depth() == 2
        assert circuit[0, 0].gate is HGate()
        assert isinstance(circuit[1, 0].gate, CircuitGate)
        test_gate: CircuitGate = circuit[1, 0].gate  # type: ignore
        assert test_gate._circuit[0, 1].gate is CNOTGate()
        assert test_gate._circuit[0, 2].gate is CNOTGate()
        assert test_gate._circuit[1, 0].gate is wide_gate
        assert test_gate._circuit[1, 1].gate is wide_gate
        assert test_gate._circuit[1, 2].gate is wide_gate
        check_no_idle_cycles(circuit)
        assert np.allclose(utry.get_numpy(), circuit.get_unitary().get_numpy())

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
        assert circuit.get_num_operations() == 9
        assert circuit.get_depth() == 3
        assert circuit.count(HGate()) == 2
        assert circuit.count(XGate()) == 6
        assert isinstance(circuit[1, 1].gate, CircuitGate)
        test_gate: CircuitGate = circuit[1, 1].gate  # type: ignore
        assert test_gate._circuit[0, 1].gate is CNOTGate()
        assert test_gate._circuit[0, 2].gate is CNOTGate()
        assert test_gate._circuit[1, 0].gate is wide_gate
        assert test_gate._circuit[1, 1].gate is wide_gate
        assert test_gate._circuit[1, 2].gate is wide_gate
        check_no_idle_cycles(circuit)
        assert np.allclose(utry.get_numpy(), circuit.get_unitary().get_numpy())

    def test_parameters(self) -> None:
        circ = Circuit(2)
        circ.append_gate(CNOTGate(), [1, 0])
        circ.append_gate(U3Gate(), [0], [0, 0, 0.23])
        circ.append_gate(CNOTGate(), [1, 0])

        before_fold = circ.get_unitary()

        circ.fold(circ.get_region([(0, 0), (1, 0), (2, 0)]))

        after_fold = circ.get_unitary()

        assert after_fold == before_fold


class TestSurround:
    """This tests circuit.surround."""

    @pytest.mark.skip
    def test_small_circuit_1(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(HGate(), 1)
        region = circuit.surround((0, 1), 2)
        assert region == CircuitRegion({0: (0, 2), 1: (0, 2)})

    @pytest.mark.skip
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
        assert region == CircuitRegion({0: (0, 1), 1: (0, 2)})

    @pytest.mark.skip
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
        region = circuit.surround((0, 1), 3)
        assert region == CircuitRegion({0: (0, 5), 1: (0, 5), 2: (0, 5)})

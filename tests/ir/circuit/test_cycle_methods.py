"""This test module verifies all circuit cycle methods."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import CPIGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import ZGate
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import valid_type_test


class TestPopCycle:
    """This tests `circuit.pop_cycle`."""

    @valid_type_test(Circuit(1).pop_cycle)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).pop_cycle)
    def test_invalid_type(self) -> None:
        pass

    @pytest.mark.parametrize('cycle_index', [-20, -10, -5, 5, 8, 10, 100])
    def test_index_invalid_1(self, cycle_index: int) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        try:
            circuit.pop_cycle(cycle_index)
        except IndexError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    @pytest.mark.parametrize('cycle_index', [-20, -10, -5, 5, 8, 10, 100])
    def test_index_invalid_2(self, cycle_index: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        try:
            circuit.pop_cycle(cycle_index)
        except IndexError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_index_valid_1(self, r6_qudit_circuit: Circuit) -> None:
        num_cycles = r6_qudit_circuit.num_cycles
        print('!' * 80)
        for i in range(num_cycles):
            print('!' * 20)
            for cycle in r6_qudit_circuit._circuit:
                print(cycle)
            r6_qudit_circuit.pop_cycle(-1)
            assert r6_qudit_circuit.num_cycles == num_cycles - i - 1
            print('!' * 20)

    @pytest.mark.parametrize('cycle_index', [-3, -2, -1, 0, 1, 2])
    def test_multi_qudit(self, cycle_index: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]),
            [0, 1, 2, 3],
        )
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]),
            [0, 1, 2, 3],
        )
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]),
            [0, 1, 2, 3],
        )
        assert circuit.num_cycles == 3
        assert circuit.num_operations == 3
        circuit.pop_cycle(cycle_index)
        assert circuit.num_cycles == 2
        assert circuit.num_operations == 2


class TestIsCycleInRange:
    """This tests `circuit.is_cycle_in_range`."""

    @valid_type_test(Circuit(1).is_cycle_in_range)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).is_cycle_in_range)
    def test_invalid_type(self) -> None:
        pass

    @given(integers())
    def test_return_type(self, val: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        assert isinstance(circuit.is_cycle_in_range(val), (bool, np.bool_))

    @pytest.mark.parametrize('cycle_index', [-5, -4, -3, -2, -1])
    def test_true_neg(self, cycle_index: int) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        assert circuit.is_cycle_in_range(cycle_index)

    @pytest.mark.parametrize('cycle_index', [0, 1, 2, 3, 4])
    def test_true_pos(self, cycle_index: int) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        assert circuit.is_cycle_in_range(cycle_index)

    @pytest.mark.parametrize('cycle_index', [-1000, -100, -8, -6])
    def test_false_neg(self, cycle_index: int) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        assert not circuit.is_cycle_in_range(cycle_index)

    @pytest.mark.parametrize('cycle_index', [5, 6, 8, 100, 1000])
    def test_false_pos(self, cycle_index: int) -> None:
        circuit = Circuit(1)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(HGate(), [0])
        assert not circuit.is_cycle_in_range(cycle_index)


class TestIsCycleUnoccupied:
    """This tests `circuit.is_cycle_unoccupied`."""

    @valid_type_test(Circuit(1).is_cycle_unoccupied)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).is_cycle_unoccupied)
    def test_invalid_type(self) -> None:
        pass

    @pytest.mark.parametrize(
        ('valid_int', 'location'),
        [
            (0, (0,)),
            (0, (0, 1, 2, 3)),
            (0, (0, 1, 2)),
            (0, (0, 2)),
        ],
    )
    def test_return_type_1(
            self, valid_int: int, location: Sequence[int],
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        assert isinstance(
            circuit.is_cycle_unoccupied(
                valid_int, location,
            ), (bool, np.bool_),
        )

    @pytest.mark.parametrize(
        ('valid_int', 'location'),
        [
            (1, (0,)),
            (2, (0, 1, 2, 3)),
            (0, (0, 1, 2)),
            (1, (0, 2)),
        ],
    )
    def test_return_type_2(
            self, valid_int: int, location: Sequence[int],
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CNOTGate(), [0, 1])
        assert isinstance(
            circuit.is_cycle_unoccupied(
                valid_int, location,
            ), (bool, np.bool_),
        )

    @pytest.mark.parametrize(
        ('cycle_index', 'location'),
        [
            (0, (2, 3)),
            (2, (2, 3)),
            (4, (0, 1)),
        ],
    )
    def test_true(
            self, cycle_index: int, location: Sequence[int],
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CPIGate(), [2, 3])
        assert circuit.is_cycle_unoccupied(cycle_index, location)

    @pytest.mark.parametrize(
        ('cycle_index', 'location'),
        [
            (0, (0, 1)),
            (2, (0, 1)),
            (4, (2, 3)),
            (0, (0,)),
            (0, (1,)),
            (1, (0,)),
            (1, (1,)),
            (1, (0, 1, 2, 3)),
        ],
    )
    def test_false(
            self, cycle_index: int, location: Sequence[int],
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CPIGate(), [2, 3])
        assert not circuit.is_cycle_unoccupied(cycle_index, location)

    def test_example(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(XGate(), [0])
        circuit.append_gate(ZGate(), [1])
        assert not circuit.is_cycle_unoccupied(0, [0])
        assert circuit.is_cycle_unoccupied(1, [1])


class TestFindAvailableCycle:
    """This tests `circuit.find_available_cycle`."""

    @valid_type_test(Circuit(1).find_available_cycle)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).find_available_cycle)
    def test_invalid_type(self) -> None:
        pass

    @pytest.mark.parametrize('location', [(0,), (0, 1, 2), (0, 2)])
    def test_return_type(self, location: Sequence[int]) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(ConstantUnitaryGate(np.identity(2), [2]), [1])
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(ConstantUnitaryGate(np.identity(3), [3]), [3])
        assert isinstance(circuit.find_available_cycle(location), int)

    def test(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        assert circuit.find_available_cycle([2, 3]) == 0
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CNOTGate(), [0, 1])
        assert circuit.find_available_cycle([2, 3]) == 2
        circuit.append_gate(
            ConstantUnitaryGate(np.identity(36), [2, 2, 3, 3]), [0, 1, 2, 3],
        )
        circuit.append_gate(CPIGate(), [2, 3])
        assert circuit.find_available_cycle([0, 1]) == 4

    def test_example(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        assert circuit.find_available_cycle([1]) == 0
        circuit.append_gate(XGate(), [0])
        circuit.append_gate(ZGate(), [1])
        assert circuit.find_available_cycle([1]) == 1

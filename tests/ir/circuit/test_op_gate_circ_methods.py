"""
This test module verifies all circuit operation, gate, and circuit methods.

Circuit operation, gate, and circuit methods:
    def is_point_in_range(self, point: CircuitPointLike) -> bool:
    def check_valid_operation(self, op: Operation) -> None:
    def get_operation(self, point: CircuitPointLike) -> Operation:
    def point(
        self,
        op: Operation | Gate,
        start: CircuitPointLike = (0, 0),
        end: CircuitPointLike | None = None,
    ) -> CircuitPoint:
    def append(self, op: Operation) -> None:
    def append_gate(
        self,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
    def append_circuit(
        self,
        circuit: Circuit,
        location: Sequence[int],
    ) -> None:
    def extend(self, ops: Iterable[Operation]) -> None:
    def insert(self, cycle_index: int, op: Operation) -> None:
    def insert_gate(
        self,
        cycle_index: int,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
    def insert_circuit(
        self,
        cycle_index: int,
        circuit: Circuit,
        location: Sequence[int],
    ) -> None:
    def remove(self, op: Operation | Gate) -> None:
    def count(self, op: Operation | Gate) -> int:
    def pop(self, point: CircuitPointLike | None = None) -> Operation:
    def batch_pop(self, points: Sequence[CircuitPointLike]) -> Circuit:
    def replace(self, point: CircuitPointLike, op: Operation) -> None:
    def replace_gate(
        self,
        point: CircuitPointLike,
        gate: Gate,
        location: Sequence[int],
        params: Sequence[float] = [],
    ) -> None:
    def replace_with_circuit(
        self,
        point: CircuitPointLike,
        circuit: Circuit,
        location: Sequence[int],
    ) -> None:
    def fold(self, points: Sequence[CircuitPointLike]) -> None:
    def copy(self) -> Circuit:
    def get_slice(self, points: Sequence[CircuitPointLike]) -> Circuit:
    def clear(self) -> None:
    def operations(self, reversed: bool = False) -> Iterator[Operation]:
    def operations_with_points(
            self,
            reversed: bool = False,
    ) -> Iterator[tuple[CircuitPoint, Operation]]:
    def operations_on_qudit(
        self,
        qudit_index: int,
        reversed: bool = False,
    ) -> Iterator[Operation]:
    def operations_on_qudit_with_points(
            self,
            qudit_index: int,
            reversed: bool = False,
    ) -> Iterator[tuple[CircuitPoint, Operation]]:
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CPIGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike


class TestIsPointInRange:
    """This tests circuit.is_point_in_range."""

    def test_is_point_in_range_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_point_in_range((an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_is_point_in_range_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_point_in_range(CircuitPoint(an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_is_point_in_range_type_valid_3(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.is_point_in_range((an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_is_point_in_range_type_valid_4(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.is_point_in_range(CircuitPoint(an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_is_point_in_range_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_point_in_range((not_an_int, 0))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_is_point_in_range_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_point_in_range(CircuitPoint(not_an_int, 0))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_is_point_in_range_type_invalid_3(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_point_in_range((not_an_int, not_an_int))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_is_point_in_range_type_invalid_4(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_point_in_range((0, not_an_int))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_is_point_in_range_return_type(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        assert isinstance(
            circuit.is_point_in_range(
                (an_int, an_int),
            ), (bool, np.bool_),
        )

    @pytest.mark.parametrize(
        'point', [
            (-5, -5),
            (-4, -4),
            (-3, -3),
            (-2, -2),
            (-1, -1),
        ],
    )
    def test_is_point_in_range_true_neg(self, point: CircuitPointLike) -> None:
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
    def test_is_point_in_range_true_pos(self, point: CircuitPointLike) -> None:
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
    def test_is_point_in_range_false_neg(
            self, point: CircuitPointLike,
    ) -> None:
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
    def test_is_point_in_range_false_pos(
            self, point: CircuitPointLike,
    ) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert not circuit.is_point_in_range(point)


class TestCheckValidOperation:
    """This tests circuit.check_valid_operation."""

    @pytest.mark.parametrize(
        ('circuit', 'op'),
        [
            (Circuit(1), Operation(HGate(), [0])),
            (Circuit(1), Operation(CNOTGate(), [0, 1])),
            (Circuit(1), Operation(CPIGate(), [2, 3])),
            (Circuit(4, [2, 2, 3, 3]), Operation(HGate(), [0])),
            (Circuit(4, [2, 2, 3, 3]), Operation(CNOTGate(), [0, 1])),
            (Circuit(4, [2, 2, 3, 3]), Operation(CPIGate(), [2, 3])),
        ],
    )
    def test_check_valid_operation_type_valid(
        self,
        circuit: Circuit,
        op: Operation,
    ) -> None:
        try:
            circuit.check_valid_operation(op)
        except TypeError:
            assert False, 'Unexpected Exception.'
        except BaseException:
            return

    @pytest.mark.parametrize(
        ('circuit', 'op'),
        [
            (Circuit(1), 'A'),
            (Circuit(1), 0),
            (Circuit(1), np.int64(1234)),
            (Circuit(4, [2, 2, 3, 3]), 'A'),
            (Circuit(4, [2, 2, 3, 3]), 0),
            (Circuit(4, [2, 2, 3, 3]), np.int64(1234)),
        ],
    )
    def test_check_valid_operation_type_invalid(
        self,
        circuit: Circuit,
        op: Operation,
    ) -> None:
        try:
            circuit.check_valid_operation(op)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_location_mismatch_1(self, qubit_gate: Gate) -> None:
        circuit = Circuit(qubit_gate.get_size())
        location = list(range(qubit_gate.get_size()))
        location[-1] += 1
        params = [0] * qubit_gate.get_num_params()
        op = Operation(qubit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_location_mismatch_2(self, qutrit_gate: Gate) -> None:
        circuit = Circuit(qutrit_gate.get_size(), qutrit_gate.get_radixes())
        location = list(range(qutrit_gate.get_size()))
        location[-1] += 1
        params = [0] * qutrit_gate.get_num_params()
        op = Operation(qutrit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_radix_mismatch_1(self, qubit_gate: Gate) -> None:
        circuit = Circuit(qubit_gate.get_size(), [3] * qubit_gate.get_size())
        location = list(range(qubit_gate.get_size()))
        params = [0] * qubit_gate.get_num_params()
        op = Operation(qubit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_radix_mismatch_2(self, qutrit_gate: Gate) -> None:
        circuit = Circuit(qutrit_gate.get_size())
        location = list(range(qutrit_gate.get_size()))
        params = [0] * qutrit_gate.get_num_params()
        op = Operation(qutrit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_valid_1(self, gate: Gate) -> None:
        circuit = Circuit(gate.get_size(), gate.get_radixes())
        location = list(range(gate.get_size()))
        params = [0] * gate.get_num_params()
        circuit.check_valid_operation(Operation(gate, location, params))

    def test_valid_2(self, gate: Gate) -> None:
        circuit = Circuit(gate.get_size() + 2, (2, 2) + gate.get_radixes())
        location = [x + 2 for x in list(range(gate.get_size()))]
        params = [0] * gate.get_num_params()
        circuit.check_valid_operation(Operation(gate, location, params))

    def test_valid_3(self) -> None:
        circuit = Circuit(2, [3, 2])
        gate = ConstantUnitaryGate(np.identity(6), [2, 3])
        circuit.check_valid_operation(Operation(gate, [1, 0]))


class TestGetOperation:
    """This tests circuit.get_operation."""

    def test_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.get_operation((an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.get_operation(CircuitPoint(an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_3(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.get_operation((an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_4(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.get_operation(CircuitPoint(an_int, an_int))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.get_operation((not_an_int, 0))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.get_operation(CircuitPoint(not_an_int, 0))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_3(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.get_operation((not_an_int, not_an_int))
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    @pytest.mark.parametrize(
        'point', [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
        ],
    )
    def test_return_type(self, point: CircuitPointLike) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert isinstance(circuit.get_operation(point), Operation)

    @pytest.mark.parametrize(
        'point', [
            (-1000, 0),
            (1, -100),
            (-8, -8),
            (-6, -6),
            (-7, 4),
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
    def test_index_error_out_of_bounds(self, point: CircuitPointLike) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        try:
            circuit.get_operation(point)
        except IndexError:
            return

        assert False, 'Should not have reached here.'

    def test_correctness_1(self, r6_qudit_circuit: Circuit) -> None:
        for x in range(r6_qudit_circuit.get_num_cycles()):
            for y in range(r6_qudit_circuit.get_size()):
                correct = r6_qudit_circuit._circuit[x][y]

                if correct is not None:
                    assert correct is r6_qudit_circuit.get_operation((x, y))

                else:
                    try:
                        r6_qudit_circuit.get_operation((x, y))
                    except IndexError:
                        pass
                    except BaseException:
                        assert False, 'Unexpected exception.'

    def test_correctness_2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])
        assert circuit.get_operation((0, 0)).gate == HGate()
        assert circuit.get_operation((1, 0)).gate == CNOTGate()
        assert circuit.get_operation((1, 1)).gate == CNOTGate()

    def test_example(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.get_operation((1, 0)).__repr__() == 'CNOTGate@(0,1)'


class TestPoint:
    """This tests circuit.point."""

    def test_type_valid_1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.point(HGate())
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.point(Operation(HGate(), [0]))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_3(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.point(HGate())
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_4(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.point(Operation(HGate(), [0]))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.point(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.point(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_return_type(self) -> None:
        circuit = Circuit(5)
        for i in range(5):
            circuit.append_gate(HGate(), [0])
            circuit.append_gate(HGate(), [1])
            circuit.append_gate(HGate(), [2])
            circuit.append_gate(HGate(), [3])
            circuit.append_gate(HGate(), [4])
        assert isinstance(circuit.point(HGate()), CircuitPoint)

    def test_correctness_1(self, r6_qudit_circuit: Circuit) -> None:
        for x in range(r6_qudit_circuit.get_num_cycles()):
            for y in range(r6_qudit_circuit.get_size()):
                op = r6_qudit_circuit._circuit[x][y]

                if op is not None:
                    point = r6_qudit_circuit.point(op, (x, y))
                    assert r6_qudit_circuit.get_operation(point) is op
                    point = r6_qudit_circuit.point(op, (x, y), (x, y))
                    assert r6_qudit_circuit.get_operation(point) is op

                    point = r6_qudit_circuit.point(op.gate, (x, y))
                    assert r6_qudit_circuit.get_operation(point) is op
                    point = r6_qudit_circuit.point(op.gate, (x, y), (x, y))
                    assert r6_qudit_circuit.get_operation(point) is op

    def test_correctness_2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])
        assert circuit.point(HGate()) == (0, 0)
        assert circuit.point(CNOTGate()) == (1, 0)
        assert circuit.point(Operation(HGate(), [0])) == (0, 0)
        assert circuit.point(Operation(CNOTGate(), [0, 1])) == (1, 0)

        try:
            circuit.point(Operation(CNOTGate(), [1, 0]))
        except ValueError:
            return
        assert False, 'Should not have reached here.'

    def test_invalid_value_1(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])

        try:
            circuit.point(CPIGate())
        except ValueError:
            return

        assert False, 'Should not have reached here.'

    def test_invalid_value_2(self) -> None:
        circuit = Circuit(2)
        circuit.append_gate(HGate(), [0])
        circuit.append_gate(CNOTGate(), [0, 1])

        try:
            circuit.point(XGate())
        except ValueError:
            return

        assert False, 'Should not have reached here.'

    def test_example(self) -> None:
        circuit = Circuit(1)

        opH = Operation(HGate(), [0])
        circuit.append(opH)
        assert circuit.point(opH).__repr__(
        ) == 'CircuitPoint(cycle=0, qudit=0)'

        opX = Operation(XGate(), [0])
        circuit.append(opX)
        assert circuit.point(opX).__repr__(
        ) == 'CircuitPoint(cycle=1, qudit=0)'


class TestAppend:
    """This tests circuit.append."""

    def test_append_type_valid_1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append(Operation(HGate(), [0]))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_type_valid_2(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append(Operation(HGate(), [2]))
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.append(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'


class TestAppendGate:
    """This tests circuit.append_gate."""

    def test_append_gate_type_valid_1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_gate(HGate(), [0])
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_gate_type_valid_2(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_gate(HGate(), [2])
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_gate_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_gate(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_gate_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_gate(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'


class TestAppendCircuit:
    """This tests circuit.append_circuit."""

    def test_append_circuit_type_valid_1(self) -> None:
        circuit = Circuit(1)
        circuit_to_add = Circuit(1)
        circuit_to_add.append_gate(HGate(), [0])
        try:
            circuit.append_circuit(circuit_to_add, [0])
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_circuit_type_valid_2(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        circuit_to_add = Circuit(1)
        circuit_to_add.append_gate(HGate(), [0])
        try:
            circuit.append_circuit(circuit_to_add, [0])
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_circuit_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_circuit(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_circuit_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_circuit(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'


class TestExtend:
    """This tests circuit.extend."""

    def test_extend_type_valid_1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend([Operation(HGate(), [0])])
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_extend_type_valid_2(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.extend([Operation(HGate(), [0])])
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_extend_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.extend(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'


class TestInsert:
    """This tests circuit.insert."""


class TestInsertGate:
    """This tests circuit.insert_gate."""


class TestInsertCircuit:
    """This tests circuit.insert_circuit."""


class TestRemove:
    """This tests circuit.remove."""


class TestCount:
    """This tests circuit.count."""


class TestPop:
    """This tests circuit.pop."""


class TestBatchPop:
    """This tests circuit.batch_pop."""


class TestReplace:
    """This tests circuit.replace."""


class TestReplaceGate:
    """This tests circuit.replace_gate."""


class TestReplaceWithCircuit:
    """This tests circuit.replace_with_circuit."""


class TestFold:
    """This tests circuit.fold."""


class TestCopy:
    """This tests circuit.copy."""


class TestGetSlice:
    """This tests circuit.get_slice."""


class TestClear:
    """This tests circuit.clear."""


class TestOperations:
    """This tests circuit.operations."""


class TestOperationsWithPoints:
    """This tests circuit.operations_with_points."""


class TestOperationsOnQudit:
    """This tests circuit.operations_on_qudit."""


class TestOperationsOnQuditWithPoints:
    """This tests circuit.operations_on_qudit_with_points."""

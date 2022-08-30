"""This test module verifies all circuit operation, gate, and circuit
methods."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import CPIGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.point import CircuitPointLike
from bqskit.utils.test.strategies import circuits
from bqskit.utils.test.strategies import operations
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import valid_type_test


def check_no_idle_cycles(circuit: Circuit) -> None:
    for cycle_index in range(circuit.num_cycles):
        assert not circuit._is_cycle_idle(cycle_index)


class TestCheckValidOperation:
    """This tests `circuit.check_valid_operation`."""

    @valid_type_test(Circuit(1).check_valid_operation)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).check_valid_operation)
    def test_invalid_type(self) -> None:
        pass

    def test_location_mismatch_1(self, qubit_gate: Gate) -> None:
        circuit = Circuit(qubit_gate.num_qudits)
        location = list(range(qubit_gate.num_qudits))
        location[-1] += 1
        params = [0] * qubit_gate.num_params
        op = Operation(qubit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_location_mismatch_2(self, qutrit_gate: Gate) -> None:
        circuit = Circuit(qutrit_gate.num_qudits, qutrit_gate.radixes)
        location = list(range(qutrit_gate.num_qudits))
        location[-1] += 1
        params = [0] * qutrit_gate.num_params
        op = Operation(qutrit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_radix_mismatch_1(self, qubit_gate: Gate) -> None:
        circuit = Circuit(qubit_gate.num_qudits, [3] * qubit_gate.num_qudits)
        location = list(range(qubit_gate.num_qudits))
        params = [0] * qubit_gate.num_params
        op = Operation(qubit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_radix_mismatch_2(self, qutrit_gate: Gate) -> None:
        circuit = Circuit(qutrit_gate.num_qudits)
        location = list(range(qutrit_gate.num_qudits))
        params = [0] * qutrit_gate.num_params
        op = Operation(qutrit_gate, location, params)
        try:
            circuit.check_valid_operation(op)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception'
        assert False

    def test_valid_1(self, gate: Gate) -> None:
        circuit = Circuit(gate.num_qudits, gate.radixes)
        location = list(range(gate.num_qudits))
        params = [0] * gate.num_params
        circuit.check_valid_operation(Operation(gate, location, params))

    def test_valid_2(self, gate: Gate) -> None:
        circuit = Circuit(gate.num_qudits + 2, (2, 2) + gate.radixes)
        location = [x + 2 for x in list(range(gate.num_qudits))]
        params = [0] * gate.num_params
        circuit.check_valid_operation(Operation(gate, location, params))

    def test_valid_3(self) -> None:
        circuit = Circuit(2, [3, 2])
        gate = ConstantUnitaryGate(np.identity(6), [2, 3])
        circuit.check_valid_operation(Operation(gate, [1, 0]))


class TestGetOperation:
    """This tests `circuit.get_operation`."""

    @valid_type_test(Circuit(1).get_operation)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).get_operation)
    def test_invalid_type(self) -> None:
        pass

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
        for x in range(r6_qudit_circuit.num_cycles):
            for y in range(r6_qudit_circuit.num_qudits):
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
    """This tests `circuit.point`."""

    @valid_type_test(Circuit(1).point)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).point, [IndexError])
    def test_invalid_type(self) -> None:
        pass

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
        for x in range(r6_qudit_circuit.num_cycles):
            for y in range(r6_qudit_circuit.num_qudits):
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
        ) == '(0, 0)'

        opX = Operation(XGate(), [0])
        circuit.append(opX)
        assert circuit.point(opX).__repr__(
        ) == '(1, 0)'


class TestAppend:
    """This tests `circuit.append`."""

    @valid_type_test(Circuit(1).append)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).append)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits())
    def test_reconstruct(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for op in circuit:
            new_circuit.append(op)
        check_no_idle_cycles(new_circuit)
        assert new_circuit.get_unitary() == circuit.get_unitary()


class TestAppendGate:
    """This tests `circuit.append_gate`."""

    @valid_type_test(Circuit(1).append_gate)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).append_gate, [ValueError])
    def test_invalid_type(self) -> None:
        pass

    @given(circuits())
    def test_reconstruct(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for op in circuit:
            new_circuit.append_gate(op.gate, op.location, op.params)
        check_no_idle_cycles(new_circuit)
        assert new_circuit.get_unitary() == circuit.get_unitary()


class TestAppendCircuit:
    """This tests `circuit.append_circuit`."""

    @given(circuits())
    def test_reconstruct(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        new_circuit.append_circuit(circuit, list(range(circuit.num_qudits)))
        check_no_idle_cycles(new_circuit)
        assert new_circuit.get_unitary() == circuit.get_unitary()

    @given(circuits())
    def test_reconstruct_larger(self, circuit: Circuit) -> None:
        new_circ = Circuit(circuit.num_qudits + 1, circuit.radixes + (2,))
        new_circ.append_circuit(circuit, list(range(circuit.num_qudits)))
        check_no_idle_cycles(new_circ)
        circuit.append_qudit()
        assert new_circ.get_unitary() == circuit.get_unitary()


class TestExtend:
    """This tests `circuit.extend`."""

    @valid_type_test(Circuit(1).extend)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).extend)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits())
    def test_reconstruct(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        new_circuit.extend(circuit)
        check_no_idle_cycles(new_circuit)
        assert new_circuit.get_unitary() == circuit.get_unitary()


class TestInsert:
    """This tests `circuit.insert`."""

    @valid_type_test(Circuit(1).insert)
    def test_valid_type(self) -> None:
        pass

    def test_empty(self) -> None:
        circuit = Circuit(2)
        circuit.insert(0, Operation(CXGate(), (0, 1)))
        assert circuit[0, 0] == Operation(CXGate(), (0, 1))

    @given(circuits((2, 2, 2, 2)), operations(2, max_qudit=3))
    def test_insert(self, circuit: Circuit, op: Operation) -> None:
        circuit.insert(0, op)
        assert circuit[0, op.location[0]] == op
        circuit.insert(circuit.num_cycles, op)
        assert circuit[-1, op.location[0]] == op
        check_no_idle_cycles(circuit)


class TestInsertGate:
    """This tests `circuit.insert_gate`."""

    @valid_type_test(Circuit(1).insert_gate)
    def test_valid_type(self) -> None:
        pass

    def test_empty(self) -> None:
        circuit = Circuit(2)
        circuit.insert_gate(0, CXGate(), (0, 1))
        assert circuit[0, 0] == Operation(CXGate(), (0, 1))

    @given(circuits((2, 2, 2, 2)), operations(2, max_qudit=3))
    def test_insert(self, circuit: Circuit, op: Operation) -> None:
        circuit.insert_gate(0, op.gate, op.location, op.params)
        assert circuit[0, op.location[0]] == op
        circuit.insert_gate(circuit.num_cycles, op.gate, op.location, op.params)
        assert circuit[-1, op.location[0]] == op
        check_no_idle_cycles(circuit)


class TestInsertCircuit:
    """This tests `circuit.insert_circuit`."""

    @given(circuits((2, 2, 2, 2)))
    def test_apply(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        location = list(range(circuit.num_qudits))
        new_circuit.insert_circuit(0, circuit, location)
        U = circuit.get_unitary()
        assert U == new_circuit.get_unitary()
        check_no_idle_cycles(circuit)

        new_circuit.insert_circuit(new_circuit.num_cycles, circuit, location)
        assert U @ U == new_circuit.get_unitary()
        check_no_idle_cycles(circuit)

        new_circuit.insert_circuit(
            0,
            circuit,
            location,
        )
        assert U @ U @ U == new_circuit.get_unitary()
        check_no_idle_cycles(circuit)


class TestRemove:
    """This tests `circuit.remove`."""

    @valid_type_test(Circuit(1).remove)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).remove)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_remove(self, circuit: Circuit) -> None:
        num_ops = circuit.num_operations
        while num_ops > 0:
            op = list(circuit.operations())[0]
            old_count = circuit.count(op)
            circuit.remove(op)
            assert num_ops - circuit.num_operations == 1
            assert old_count - circuit.count(op) == 1
            num_ops = circuit.num_operations
            check_no_idle_cycles(circuit)


class TestRemoveAll:
    """This tests `circuit.remove_all`."""

    @valid_type_test(Circuit(1).remove_all)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).remove_all)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_remove_all_op(self, circuit: Circuit) -> None:
        num_ops = circuit.num_operations
        while num_ops > 0:
            op = list(circuit.operations())[0]
            old_count = circuit.count(op)
            circuit.remove_all(op)
            assert num_ops - circuit.num_operations == old_count
            assert circuit.count(op) == 0
            with pytest.raises((ValueError, IndexError)):
                circuit.point(op)
            num_ops = circuit.num_operations
            check_no_idle_cycles(circuit)

    @given(circuits((2, 2, 2, 2)))
    def test_remove_all_gate(self, circuit: Circuit) -> None:
        num_ops = circuit.num_operations
        while num_ops > 0:
            op = list(circuit.operations())[0]
            old_count = circuit.count(op.gate)
            circuit.remove_all(op.gate)
            assert num_ops - circuit.num_operations == old_count
            assert circuit.count(op.gate) == 0
            with pytest.raises((ValueError, IndexError)):
                circuit.point(op.gate)
            num_ops = circuit.num_operations
            check_no_idle_cycles(circuit)


class TestCount:
    """This tests `circuit.count`."""

    @valid_type_test(Circuit(1).count)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).count)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_count_op(self, circuit: Circuit) -> None:
        for op in circuit:
            count = circuit.count(op)
            start = (0, 0)
            for i in range(count):
                start = circuit.point(op, start)
                start = (start[0] + 1, 0)

            with pytest.raises((ValueError, IndexError)):
                circuit.point(op, start)

    @given(circuits((2, 2, 2, 2)))
    def test_count_gate(self, circuit: Circuit) -> None:
        for op in circuit:
            count = circuit.count(op.gate)
            assert count == len([op2 for op2 in circuit if op2.gate == op.gate])


class TestPop:
    """This tests `circuit.pop`."""

    @valid_type_test(Circuit(1).pop)
    def test_valid_type(self) -> None:
        pass

    @invalid_type_test(Circuit(1).pop)
    def test_invalid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_pop_all(self, circuit: Circuit) -> None:
        for x in range(circuit.num_operations):
            assert isinstance(circuit.pop(), Operation)
            check_no_idle_cycles(circuit)
        assert circuit.num_operations == 0


class TestBatchPop:
    """This tests `circuit.batch_pop`."""

    @given(circuits((2, 2, 2, 2)))
    def test_batch_pop_all(self, circuit: Circuit) -> None:
        if circuit.num_operations == 0:
            return
        pts = [
            (x, y)
            for x in range(circuit.num_cycles)
            for y in range(circuit.num_qudits)
        ]
        popped_circuit = circuit.batch_pop(pts)
        assert isinstance(popped_circuit, Circuit)
        check_no_idle_cycles(popped_circuit)
        assert circuit.num_operations == 0


class TestReplace:
    """This tests `circuit.replace`."""

    @valid_type_test(Circuit(1).replace)
    def test_valid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_replace(self, circuit: Circuit) -> None:
        if circuit.num_operations == 0:
            return
        op = list(circuit.operations())[0]
        point = circuit.point(op)
        U = circuit.get_unitary()
        circuit.replace(point, op)
        assert circuit.get_unitary() == U


class TestBatchReplace:
    """This tests `circuit.replace`."""

    @valid_type_test(Circuit(1).batch_replace)
    def test_valid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_batch_replace(self, circuit: Circuit) -> None:
        ops = list(circuit.operations())
        ops = ops[:1] if len(ops) > 2 else ops
        points = [circuit.point(op) for op in ops]
        U = circuit.get_unitary()
        circuit.batch_replace(points, ops)
        assert circuit.get_unitary() == U


class TestReplaceGate:
    """This tests `circuit.replace_gate`."""

    @valid_type_test(Circuit(1).replace)
    def test_valid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_replace(self, circuit: Circuit) -> None:
        if circuit.num_operations == 0:
            return
        op = list(circuit.operations())[0]
        point = circuit.point(op)
        U = circuit.get_unitary()
        circuit.replace_gate(point, op.gate, op.location, op.params)
        assert circuit.get_unitary() == U


class TestReplaceWithCircuit:
    """This tests `circuit.replace_with_circuit`."""

    @valid_type_test(Circuit(1).replace)
    def test_valid_type(self) -> None:
        pass

    @given(circuits((2, 2, 2, 2)))
    def test_replace(self, circuit: Circuit) -> None:
        if circuit.num_operations == 0:
            return
        op = list(circuit.operations())[0]
        circ = Circuit.from_operation(op)
        point = circuit.point(op)
        U = circuit.get_unitary()
        circuit.replace_with_circuit(point, circ)
        assert circuit.get_unitary() == U


class TestCopy:
    """This tests `circuit.copy`."""

    @given(circuits((2, 2, 2, 2)))
    def test_copy(self, circuit: Circuit) -> None:
        new_circuit = circuit.copy()
        new_circuit.get_unitary() == circuit.get_unitary()


class TestBecome:
    """This tests `circuit.copy`."""

    @given(circuits((2, 2, 2, 2)))
    def test_become(self, circuit: Circuit) -> None:
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        new_circuit.become(circuit)
        new_circuit.get_unitary() == circuit.get_unitary()


class TestClear:
    """This tests `circuit.clear`."""

    @given(circuits((2, 2, 2, 2)))
    def test_clear(self, circuit: Circuit) -> None:
        circuit.clear()
        assert circuit.num_operations == 0
        assert len(circuit.gate_set) == 0
        assert circuit.depth == 0
        assert circuit.parallelism == 0
        assert circuit.num_cycles == 0
        assert len(circuit.active_qudits) == 0

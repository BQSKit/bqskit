"""This module tests circuit simulation through the get_unitary method."""
from __future__ import annotations

from random import randint

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.operation import Operation
from bqskit.passes.partitioning.scan import ScanPartitioner


class TestBatchReplace:
    def test_batch_replace(self) -> None:
        circ = Circuit(4)
        op_1a = Operation(CNOTGate(), [0, 1])
        op_2a = Operation(CNOTGate(), [2, 3])
        op_3a = Operation(CNOTGate(), [1, 2])
        op_4a = Operation(CNOTGate(), [0, 1])
        op_5a = Operation(CNOTGate(), [0, 1])
        op_6a = Operation(CNOTGate(), [2, 3])
        list_a = [op_1a, op_2a, op_3a, op_4a, op_5a, op_6a]

        op_1b = Operation(CNOTGate(), [1, 0])
        op_2b = Operation(CNOTGate(), [3, 2])
        op_3b = Operation(CNOTGate(), [2, 1])
        op_4b = Operation(CNOTGate(), [1, 0])
        op_5b = Operation(CNOTGate(), [1, 0])
        op_6b = Operation(CNOTGate(), [3, 2])
        list_b = [op_1b, op_2b, op_3b, op_4b, op_5b, op_6b]

        for op in list_a:
            circ.append(op)

        assert circ.get_operation(
            (0, 0),
        ) == op_1a and circ.get_operation(
            (0, 1),
        ) == op_1a
        assert circ.get_operation(
            (0, 2),
        ) == op_2a and circ.get_operation(
            (0, 3),
        ) == op_2a

        assert circ.get_operation(
            (1, 1),
        ) == op_3a and circ.get_operation(
            (1, 2),
        ) == op_3a

        assert circ.get_operation(
            (2, 0),
        ) == op_4a and circ.get_operation(
            (2, 1),
        ) == op_4a
        assert circ.get_operation(
            (2, 2),
        ) == op_6a and circ.get_operation(
            (2, 3),
        ) == op_6a

        assert circ.get_operation(
            (3, 0),
        ) == op_5a and circ.get_operation(
            (3, 1),
        ) == op_5a
        for i in range(4):
            for j in range(4):
                print(f'({i},{j}): {circ._circuit[i][j]}')

        points = [(0, 0), (0, 2), (1, 1), (2, 0), (3, 1), (2, 3)]
        new_ops = list_b
        circ.batch_replace(points, new_ops)

        assert circ.get_operation(
            (0, 0),
        ) == op_1b and circ.get_operation(
            (0, 1),
        ) == op_1b
        assert circ.get_operation(
            (0, 2),
        ) == op_2b and circ.get_operation(
            (0, 3),
        ) == op_2b

        assert circ.get_operation(
            (1, 1),
        ) == op_3b and circ.get_operation(
            (1, 2),
        ) == op_3b

        assert circ.get_operation(
            (2, 0),
        ) == op_4b and circ.get_operation(
            (2, 1),
        ) == op_4b
        assert circ.get_operation(
            (2, 2),
        ) == op_6b and circ.get_operation(
            (2, 3),
        ) == op_6b

        assert circ.get_operation(
            (3, 0),
        ) == op_5b and circ.get_operation(
            (3, 1),
        ) == op_5b

    def test_random_batch_replace(self) -> None:
        num_gates = 200
        num_q = 10

        circ = Circuit(num_q)
        for _ in range(num_gates):
            a = randint(0, num_q - 1)
            b = randint(0, num_q - 1)
            if a == b:
                b = (b + 1) % num_q
            circ.append_gate(CNOTGate(), [a, b])

        ScanPartitioner(2).run(circ, {})

        points = []
        ops = []
        for cycle, op in circ.operations_with_cycles():
            point = (cycle, op.location[0])
            ops.append(Operation(CNOTGate(), op.location))
            points.append(point)

        circ.batch_replace(points, ops)

        for op in circ:
            assert isinstance(op.gate, CNOTGate)
        # Because pops are used, there is no guarantee that gate will not shift

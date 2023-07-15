from __future__ import annotations

from bqskit.ir import Operation
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.passes import GroupSingleQuditGatePass


def test_single_qudit_grouper(r6_qudit_circuit: Circuit) -> None:
    r6_qudit_circuit.perform(GroupSingleQuditGatePass())

    # All single-qudit gates should be in a CircuitGate
    for gate in r6_qudit_circuit.gate_set:
        if gate.num_qudits == 1:
            assert isinstance(gate, CircuitGate)

    # There should be no two single-qudit CircuitGates next to one another
    def is_single_qudit_block(op: Operation) -> bool:
        return isinstance(op, CircuitGate) and op.num_qudits == 1

    for qudit in range(r6_qudit_circuit.num_qudits):
        prev_op = None
        for op in r6_qudit_circuit.operations(qudits_or_region=[qudit]):
            if prev_op is None:
                prev_op = op
                continue

            if is_single_qudit_block(prev_op) and is_single_qudit_block(op):
                assert False, 'Consecutive single-qudit blocks.'
            prev_op = op

from __future__ import annotations

from bqskit.compiler.passes.partitioning.greedy import GreedyPartitioner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import TaggedGate


class TestGreedyPartitioner:

    def test_run_r6(self, r6_qudit_circuit: Circuit) -> None:
        utry = r6_qudit_circuit.get_unitary()
        GreedyPartitioner(3).run(r6_qudit_circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in r6_qudit_circuit
        )
        assert all(
            not isinstance(op.gate, TaggedGate)
            or not op.gate.tag != '__fold_placeholder__'
            for op in r6_qudit_circuit
        )
        assert r6_qudit_circuit.get_unitary() == utry
        for cycle_index in range(r6_qudit_circuit.get_num_cycles()):
            assert not r6_qudit_circuit._is_cycle_idle(cycle_index)

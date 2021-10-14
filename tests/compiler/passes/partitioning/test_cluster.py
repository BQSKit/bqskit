from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.passes.partitioning.cluster import ClusteringPartitioner


class TestClusteringPartitioner:

    def test_run_r6(self, r6_qudit_circuit: Circuit) -> None:
        utry = r6_qudit_circuit.get_unitary()
        ClusteringPartitioner(3, 2).run(r6_qudit_circuit, {})

        assert any(
            isinstance(op.gate, CircuitGate)
            for op in r6_qudit_circuit
        )

        assert r6_qudit_circuit.get_unitary() == utry
        for cycle_index in range(r6_qudit_circuit.num_cycles):
            assert not r6_qudit_circuit._is_cycle_idle(cycle_index)

from __future__ import annotations

from bqskit.compiler.passes.partitioning.scan import ScanPartitioner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates import TaggedGate


class TestScanPartitioner:

    def test_run(self) -> None:
        """Test run with a linear topology."""
        #     0  1  2  3  4        #########
        # 0 --o-----o--P--P--    --#-o---o-#-----#######--
        # 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
        # 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
        # 3 --o--P--x--P--x--    --#########-o-x-#---x-#--
        # 4 --x-----------P--    ----------#-x---#######--
        #                                  #######

        num_q = 5
        circ = Circuit(num_q)
        circ.append_gate(CNOTGate(), [0, 1])
        circ.append_gate(CNOTGate(), [3, 4])
        circ.append_gate(CNOTGate(), [1, 2])
        circ.append_gate(CNOTGate(), [0, 1])
        circ.append_gate(CNOTGate(), [2, 3])
        circ.append_gate(CNOTGate(), [1, 2])
        circ.append_gate(CNOTGate(), [2, 3])
        utry = circ.get_unitary()
        ScanPartitioner(3).run(circ, {})

        assert len(circ) == 3
        assert all(isinstance(op.gate, CircuitGate) for op in circ)
        placeholder_gate = TaggedGate(IdentityGate(1), '__fold_placeholder__')
        assert all(op.gate._circuit.count(placeholder_gate) == 0 for op in circ)  # type: ignore  # noqa
        assert circ.get_unitary() == utry
        for cycle_index in range(circ.get_num_cycles()):
            assert not circ._is_cycle_idle(cycle_index)

    def test_run_r6(self, r6_qudit_circuit: Circuit) -> None:
        utry = r6_qudit_circuit.get_unitary()
        ScanPartitioner(3).run(r6_qudit_circuit, {})

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

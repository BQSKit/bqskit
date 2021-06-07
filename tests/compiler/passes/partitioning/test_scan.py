from __future__ import annotations

from bqskit.compiler.passes.partitioning.scan import ScanPartitioner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate


class TestScanPartitioner:

    def test_run(self) -> None:
        """Test run with a linear topology."""
        #     0  1  2  3  4        #########
        # 0 --o-----o--------    --#-o---o-#-----#######--
        # 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
        # 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
        # 3 --o-----x-----x--    --#########-o-x-#---x-#--
        # 4 --x--------------    ----------#-x---#######--
        #                                  #######
        num_q = 5
        # coup_map = {(0, 1), (1, 2), (2, 3), (3, 4)}
        circ = Circuit(num_q)
        circ.append_gate(CNOTGate(), [0, 1])
        circ.append_gate(CNOTGate(), [3, 4])
        circ.append_gate(CNOTGate(), [1, 2])
        circ.append_gate(CNOTGate(), [0, 1])
        circ.append_gate(CNOTGate(), [2, 3])
        circ.append_gate(CNOTGate(), [1, 2])
        circ.append_gate(CNOTGate(), [2, 3])
        part = ScanPartitioner(3)

        part.run(circ, {})

        assert len(circ) == 3

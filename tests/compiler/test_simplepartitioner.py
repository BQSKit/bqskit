from __future__ import annotations

import pytest
import itertools as it

from bqskit.compiler.passes.simplepartitioner import SimplePartitioner
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.cx import CNOTGate

class TestMachineConstructor:

    def test_constructor(self) -> None:
        """
        Test if the constructor properly sets instance variables.
        """
        mach = MachineModel(3)
        part = SimplePartitioner(mach, 3)
        assert part.block_size == 3
        assert part.num_qudits == mach.num_qudits

        
    def test_get_qudit_groups(self) -> None:
        """
        Ensure that groups found by get_qudit_groups consist of valid edges
        in some coupling graph.
        """
        num_qudits = 9
        block_size = 3
        # TEST ALL TO ALL
        mach = MachineModel(num_qudits)
        part = SimplePartitioner(mach, block_size)
        # Get all qubit groups
        groups = part.get_qudit_groups()
        for group in groups:
            # Get all combinations of edges in the group
            perms = it.combinations(group, 2)
            # Make sure the edge exists in the coupling graph
            for perm in perms:
                assert perm in part.machine.coupling_graph \
                    or (perm[1], perm[0]) in part.machine.coupling_graph
        # TEST NEAREST NEIGHBOR
        coup_map = set([(0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
            (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)])
        mach = MachineModel(num_qudits, coup_map)
        part = SimplePartitioner(mach, block_size)
        # Get all qubit groups
        groups = part.get_qudit_groups()
        for g in groups:
            # For every permutation of vertices in a group, only 2 edges should
            # actually be present in the coupling graph.
            assert len(g) == 3
            perms = it.combinations(g, 2)
            count = 0
            for perm in perms:
                if perm in part.machine.coupling_graph \
                    or (perm[1], perm[0]) in part.machine.coupling_graph:
                    count += 1
            # Edge count should always be 2
            assert count == 2


    def test_get_used_qudit_set(self) -> None:
        """
        Ensure that qudits are properly counted as idle or not.
        """
        circ = Circuit(4)
        mach = MachineModel(4)
        part = SimplePartitioner(mach)
        used_qudits = part.get_used_qudit_set(circ)
        assert len(used_qudits) == 0

        circ.append_gate(HGate(), [0])
        used_qudits = part.get_used_qudit_set(circ)
        assert used_qudits == set([0])

        for i in range(4):
            circ.append_gate(HGate(), [i])
        used_qudits = part.get_used_qudit_set(circ)
        assert used_qudits == set([0,1,2,3])


    def test_num_ops_left(self) -> None:
        """
        Ensure that the number of operations given a point in the circuit
        are properly counted.
        """
        mach = MachineModel(5)
        part = SimplePartitioner(mach)

        # 0 --H--H--H--H--
        # 1 --H--H--H-----
        # 2 --H--o--------
        # 3 --H--x--H-----
        # 4 --------------
        circ = Circuit(5)
        for i in range(4):
            circ.append_gate(HGate(), [0])
        for i in range(3):
            circ.append_gate(HGate(), [1])
        circ.append_gate(HGate(), [2])
        circ.append_gate(HGate(), [3])
        circ.append_gate(CNOTGate(), [2,3])
        circ.append_gate(HGate(), [3])

        # Make sure circuit is the right length
        assert circ.get_depth() == 4

        # Check qudit 0
        for cycle in range(circ.get_depth()):
           assert part.num_ops_left(circ, 0, cycle) == 4 - cycle
        # Check qudit 1
        for cycle in range(circ.get_depth()):
            if cycle <= 3:
                assert part.num_ops_left(circ, 1, cycle) == 3 - cycle
            else:
                assert part.num_ops_left(circ, 1, cycle) == 0
        # Check qudit 2
        for cycle in range(circ.get_depth()):
            if cycle <= 2:
                assert part.num_ops_left(circ, 2, cycle) == 2 - cycle
            else:
                assert part.num_ops_left(circ, 2, cycle) == 0
        # Check qudit 3
        for cycle in range(circ.get_depth()):
            if cycle <= 3:
                assert part.num_ops_left(circ, 3, cycle) == 3 - cycle
            else:
                assert part.num_ops_left(circ, 3, cycle) == 0
        # Check qudit 4
        for cycle in range(circ.get_depth()):
            assert part.num_ops_left(circ, 4, cycle) == 0
        
        # Make sure out of bounds references work
        assert part.num_ops_left(circ, 0, 100) == 0


    def test_subcircuititerator(self) -> None:
        # 0 --o-----o----- 
        # 1 --x--o--x--o-- 
        # 2 -----x--o--x-- 
        # 3 --o-----x----- 
        # 4 --x----------- 
        #     0  1  2  3
        circ = Circuit(5)
        mach = MachineModel(5)
        circ.append_gate(CNOTGate(), [0,1])
        circ.append_gate(CNOTGate(), [1,2])
        circ.append_gate(CNOTGate(), [0,1])
        circ.append_gate(CNOTGate(), [3,4])
        circ.append_gate(CNOTGate(), [2,3])
        circ.append_gate(CNOTGate(), [1,2])
        part = SimplePartitioner(mach)

        subiter = circ.SubCircuitIterator(
            circuit = circ._circuit,
            subset = [i for i in range(5)],
            and_points = True
        )
        for point, op in subiter:
            assert 'cx' in op.get_qasm()
            if point.cycle == 0:
                assert point.qudit in [0,1,3,4]
            elif point.cycle == 1 or point.cycle == 3:
                assert point.qudit in [1,2]
            elif point.cycle == 2:
                assert point.qudit in [0,1,2,3]

        subiter = circ.SubCircuitIterator(
            circuit = circ._circuit,
            subset = [0,3],
            and_points = True
        )
        for point, op in subiter:
            assert 'cx' in op.get_qasm()
            if point.cycle == 0:
                assert point.qudit in [0,1,3,4]
            elif point.cycle == 1 or point.cycle == 3:
                assert False
            elif point.cycle == 2:
                assert point.qudit in [0,1,2,3]

        subiter = circ.SubCircuitIterator(
            circuit = circ._circuit,
            subset = [1,2],
            and_points = True
        )
        for point, op in subiter:
            assert 'cx' in op.get_qasm()
            if point.cycle == 0:
                assert point.qudit in [0,1,3,4]
            elif point.cycle == 1 or point.cycle == 3:
                assert point.qudit in [1,2]
            elif point.cycle == 2:
                assert point.qudit in [0,1,2,3]

        subiter = circ.SubCircuitIterator(
            circuit = circ._circuit,
            subset = [4],
            and_points = True
        )
        assert circ.get_operation((0,4)) is not None
        for point, op in subiter:
            print(str(point.cycle) + " - " + str(point.qudit))
            print(subiter.max_qudit)
            assert 'cx' in op.get_qasm()
            if point.cycle == 0:
                assert point.qudit in [4]
            else:
                assert False


    def test_run(self) -> None:
        """
        Test run with a linear topology.
        """
        #     0  1  2  3  4        #########     
        # 0 --o-----o--------    --#-o---o-#-----#######--
        # 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
        # 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
        # 3 --o-----x-----x--    --#########-o-x-#---x-#--
        # 4 --x--------------    ----------#-x---#######--
        #                                  #######
        num_q = 5
        coup_map = set([(0,1),(1,2),(2,3),(3,4)])
        circ = Circuit(num_q)
        circ.append_gate(CNOTGate(), [0,1])
        circ.append_gate(CNOTGate(), [3,4])
        circ.append_gate(CNOTGate(), [1,2])
        circ.append_gate(CNOTGate(), [0,1])
        circ.append_gate(CNOTGate(), [2,3])
        circ.append_gate(CNOTGate(), [1,2])
        circ.append_gate(CNOTGate(), [2,3])
        mach = MachineModel(num_q, coup_map)
        part = SimplePartitioner(mach, 3)

        data = {}

        part.run(circ, data)

        assert len(circ) == 3

        circ_iter = circ.CircuitIterator(
            circuit = circ._circuit,
            and_points = True
        )
        for point, op in circ_iter:
            if point.cycle == 0:
                assert point.qudit in [0,1,2]
            elif point.cycle == 1:
                assert point.qudit in [2,3,4]
            elif point.cycle == 2:
                assert point.qudit in [1,2,3]
            
            assert len(op.location) == 3
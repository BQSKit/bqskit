from __future__ import annotations

import pytest
import itertools as it

from bqskit.compiler.passes.simplepartitioner import SimplePartitioner
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.h import HGate

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
        coup_map = [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
            (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)]
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

        # qudit 0 - 4 gates
        # qudit 1 - 3 gates
        # qudit 2 - 1 gate
        # qudit 3 - 2 gates
        # qudit 4 - 0 gates
        circ = Circuit(5)
        for i in range(4):
            circ.append_gate(HGate(), [0])
        for i in range(3):
            circ.append_gate(HGate(), [1])
        for i in range(2):
            circ.append_gate(HGate(), [3])
        for i in range(1):
            circ.append_gate(HGate(), [2])

        assert circ.get_depth() == 4
        for cycle in range(4):
           assert part.num_ops_left(circ, 0, cycle) == 3 - (cycle)
        

    def test_qudit_group_search(self) -> None:
        pass

    def test_set_run_parameters(self) -> None:
        pass

    def test_run(self) -> None:
        pass
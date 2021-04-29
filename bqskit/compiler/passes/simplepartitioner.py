"""This module defines the SimplePartitioner pass."""
from __future__ import annotations

from typing import Any, Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.point import CircuitPoint

import os

from math import sqrt, ceil
from operator import attrgetter

# TODO:
#   Add remapping phase for algorithm to physical qubits

class SimplePartitioner(BasePass):
    
    def __init__(
        self,
        machine: MachineModel,
        b_size: int = 3
    ) -> None:
        """
        Constructor for a SimplePartitioner based on the QGo algorithm.

        Args:
            b_size (int): Size of synthesizable partition blocks.
            machine (MachineModel): Description of the physical layout of
                qudits in hardware. This variable will set the coupling map
                used for synthesis and determines the number of physical 
                qudits in the topology.
        """
        # NOTE: num_q is the number of qudits in the physical topology, not 
        # in the algorithm/circuit.
        self.b_size = b_size
        self.machine = machine
        self.num_q = self.machine.num_qudits or 0
        self.used_qudits = set()
        self.qubit_groups = []

        # Default scores for multi qudit and single qudit gates in QGo
        self.multi_gate_score  = 1000
        self.single_gate_score = 1
        

    def get_used_qudit_set(circuit: Circuit) -> set[int]:
        """
        Return a set of the used qudits in circuit.

        Args:
            circuit (Circuit): The circuit to be analyzed.

        Returns:
            used_qudits (set[int]): The set containing the indices of all 
                qudits used in any operation during the circuit.
        """
        used_qudits = set()
        for qudit in range(circuit.size):
            if not is_qudit_idle(qudit):
                used_qudits.add(qudit)
        return used_qudits

    def get_qudit_groups(self) -> Sequence[Sequence[int]]:
        """
        Returns a list of all the valid qudit groups in the coupling map. Do
        a breadth first search on all pairs of qudits, only keep paths that 
        have length >= 2 and <= b_size. Note that the coupling map is assumed
        to be undirected.

        Args:
            None 

        Returns:
            qudit_groups (Sequence[Sequence[int]]): A list of all groups of 
                physically connected qudits that are < b_size away from each
                other.
        """
        # Create an adjaceny dict
        adj_list = []
        for i in range(0, self.num_q):
            adj_list.append([])
        for edge in self.machine.coupling_graph:
            adj_list[edge[0]].append(edge[1])
            adj_list[edge[1]].append(edge[0])
        qudits = list(range(0, self.num_q))
        adj_dict = {k: adj_list[k] for k in qudits}

        found_paths = set([])
        # For each qudit
        for s in range(0, self.num_q):
            path = set([])
            # Get every vald set containing s with cardinality == b_size
            self._qudit_group_search(adj_dict, found_paths, path, s, \
                self.b_size)

        # Return the set as a list of paths/qudit groups
        list_of_paths = []
        for path in found_paths:
            list_of_paths.append(list(path))
        return list_of_paths

    def _qudit_group_search(
        self, 
        adj_dict : dict[int,int],
        all_paths : set[frozenset[int]],
        path : set[int],
        vertex : int,
        limit  : int,
    ) -> None:
        """
        Add paths of length == limit to the all_paths list.

        Args:
            adj_dict (dict[int,int]): Adjacency list/dictionary for the graph.
            all_paths (set[frozenset[int]]): A list that countains all paths 
                found so far of length == limit.
            path (set[int]): The list that charts the current path
                through the graph.
            vertex (int): The vertex in the graph currently being examined.
            limit (int): The desired length of paths in the all_paths list.
        """
        if vertex not in path:
            curr_path = path.copy()
            curr_path.add(vertex)
            if len(curr_path) == limit:
                all_paths.add(frozenset(curr_path))
            else:
                frontier = [adj_dict[node] for node in curr_path]
                for neighbor in frontier:
                    self._qudit_group_search(adj_dict, all_paths, path, \
                        neighbor, limit)

    class BlockInfo:
        """
        The BlockInfo Class.

        A BlockInfo object contains the description of a synthesizable block.
        """
        def __init__(
            self, 
            qudits: Sequence[int] | None = None,
            score: int = 0
        ) -> None:
            """
            The BlockInfo Constructor.

            Args:
                qudits (Sequence[int]): A list of qudits in the block.
            """
            if qudits is not None:
                self.block_start = {k: 0 for k in qudits}
                self.block_end   = {k: 0 for k in qudits}
            else:
                self.block_start = {}
                self.block_end = {}
            self.score = score
            # NOTE: for each qudit, the block_start is inclusive and the
            # block_end is exclusive, meaning the block_end cycle should not
            # be included in the synthesizable block.

    def _set_run_parameters(
        self, 
        circuit: Circuit, d
        ata: dict[str, Any]
    ) -> None:
        """Set up the SimplePartitioner variables for the current run call.

        Args:
            circuit (Circuit): Circuit to be partitioned.
            data (dict[str, Any]): Additional data for the run.
        """
        # num_q is the number of physical qubits in the topology
        self.num_q = data["num_q"] if "num_q" in data else self.num_q
        # num_verts is the number of qubits used in the algorithm/circuit
        self.num_verts = circuit.size
        # Cannot have more algorithmic qubits than physical qubits
        if self.num_q < self.num_verts:
            raise ValueError(
                'Expected num_q > circuit.size, got %d (num_q) and %d'
                %(self.num_q, self.num_verts)
            )

        # If a coupling map is provided in the data dict, it will be used. 
        # Otherwise the coupling graph assigned to the MachineModel will be
        # used. 
        self.coupling_graph = data["coupling_graph"] if "coupling_graph" in 
            data else self.machine.coupling_graph

        self.b_size = data["b_size"] if "b_size" in data else self.b_size
        if self.b_size < 2 or self.b_size > self.num_q:
            raise ValueError(
                'Expected  2 <= b_size <= num_q, got %d' %(self.b_size)
            )

        # Change scores for multi and single qudit gates if desired
        self.multi_gate_score = (data["multi_gate_score"] if "multi_gate"
            "_score" in data else self.multi_gate_score)
            self.multi_gate_score
        self.single_gate_score = (data["single_gate_score"] if "single_gate"
            "_score" in data else self.single_gate_score)

        # Get the set of all vertices used in the algorithm
        #self.used_verts = get_used_qudit_set(circuit)

    def num_ops_left(
        self,
        circuit : Circuit,
        point: CircuitPoint 
    ) -> int:
        """
        Return the number of operations on a qudit at and beyond the cycle
        specified by the point given.

        Args:
            circuit (Circuit): Circuit to be analyzed.
            point (CircuitPoint): Point at and beyond which to give the number
                of operations.

        Returns:
            int: The number of operations at and beyond the given point.
        """
        qudit_iter = circuit.operations_on_qudit_with_points(point.qudit)
        num_ops = 0
        for circ_point, circ_op in qudit_iter:
            if circ_point.cycle < point.cycle:
                continue
            else:
                num_ops += 1
        return num_op

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Block gates into CircuitGates. Given the b_size, partition a 
        circuit into a series of CircuitGates (subcircuits represented as 
        immutable gates). Synthesis should be run on each of the CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.
            data (dict[str,Any]): 

        Raises:
            ValueError: If the number of qudits in the circuit is larger than
                the number of qudits in the coupling map.
            ValueError: If the blocksize is too big or too small.
        """
        self._set_run_parameters(circuit, data)

        # TODO: RENUMBERING PHASE

        # Find all paths between any two used vertices that is less than the
        # synthesizable blocksize
        self.qudit_groups = self.get_qudit_groups()

        # Do while there are still gates to partition
        # NOTE: This assumes circuit and topology qudit numbers are equal
        sched_depth = [CircuitPoint(0, i) for i in range(0, self.num_q)]
        # NOTE: May need to add circuit method that checks if there are any
        #       operations on a given qudit after a certain cycle, because the
        #       iterator does not give a point to the end of the circuit if 
        #       there is no operation there.
        while not all([self.num_ops_left(circuit, p) == 0 for p in sched_depth]):
            # Find the scores of the qudit groups.
            best_block = self.BlockInfo()
            for q_group in self.qudit_groups:
                # Examine each qudit in the group
                curr_block = self.BlockInfo(q_group)
                # Earliest cycle not partitioned is the scheduled depth
                for qudit in q_group:
                    curr_block.block_start[qudit] = sched_depth[qudit].cycle

                # Find the earliest cycle in the group in which each q is 
                # involved with an "outsider" qudit 
                insider_qudits = set(q_group)
                circ_iter = CircuitIterator(circuit, and_points = True)
                for point, op in circ_iter:
                    # Skip operations that are before the block_start or do not
                    # involve qudits in the q_group
                    if point.qudit not in q_group or \
                        point.cycle < curr_block.block_start[point.qudit]:
                        continue

                    # Check that op doesn't interact with outsider qudits
                    for other_qudit in op.location:
                        # If a qudit interacts with an outsider, mark the end
                        # of the block for it and add it to the outsiders
                        if other_qudit not in insider_qudits:
                            curr_block.block_end[point.qudit] = point.cycle
                            insider_qudits -= set([point.qudit])
                        # Else add to the score of the current block
                        else:
                            if len(op.localtion) >= 2:
                                curr_block.score += self.multi_gate_score
                            else:
                                curr_block.score += self.single_gate_score

                # Update the best_block if the current block scores higher
                if curr_block.score > best_block.score:
                    best_block = curr_block

            # Replace the highest scoring block with a CircuitGate.
            qudits = best_block.block_start.keys()
            points_in_block = []
            for q in qudits:
                points = [CircuitPoint(cycle=cyc ,qudit=q) for cyc in \
                    range(best_block.block_start[q], best_block.block_end[q])]
                for p in points:
                    points_in_block.append(p)
            circuit.fold(points_in_block)

            # Update the scheduled depth list
            for q in qudits:
                sched_depth[q].cycle = best_block.block_end[q]
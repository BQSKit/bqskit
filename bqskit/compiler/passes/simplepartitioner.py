"""This module defines the SimplePartitioner pass."""
from __future__ import annotations

from typing import Any, Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.point import CircuitPoint
from bqskit.ir.operation import Operation
from bqskit.compiler.machine import MachineModel

import os

from math import sqrt, ceil
from operator import attrgetter

# TODO:
#   Layout should be a separate pass from partitioning. The partitioner may
#   need to be changed so that it can accept a layout assignment, but by
#   default assumes the numberings in the algorithm and topology are equal.

class SimplePartitioner(BasePass):
    # Class variables
    used_qudits = set()     # set[int]
    qudit_groups = []       # list[set[int]]
    
    def __init__(
        self,
        machine: MachineModel,
        block_size: int = 3
    ) -> None:
        """
        Constructor for a SimplePartitioner based on the QGo algorithm.

        Args:
            machine (MachineModel): Description of the physical layout of
                qudits in hardware. This variable will set the coupling map
                used for synthesis and determines the number of physical 
                qudits in the topology.

            block_size (int): Size of synthesizable partition blocks.
        
        Reference:
            https://arxiv.org/pdf/2012.09835.pdf
        """
        # NOTE: num_qudits is the number of qudits in the physical topology, not 
        # in the algorithm/circuit.
        self.block_size = block_size
        self.machine = machine
        self.num_qudits = self.machine.num_qudits

        # Default scores for multi qudit and single qudit gates in QGo
        self.multi_gate_score  = 1000
        self.single_gate_score = 1
        

    def get_used_qudit_set(self, circuit: Circuit) -> set[int]:
        """
        Return a set of the used qudits in circuit.

        Args:
            circuit (Circuit): The circuit to be analyzed.

        Returns:
            used_qudits (set[int]): The set containing the indices of all 
                qudits used in any operation during the circuit.
        """
        used_qudits = set()
        for qudit in range(circuit.get_size()):
            if not circuit.is_qudit_idle(qudit):
                used_qudits.add(qudit)
        return used_qudits

    def get_qudit_groups(self) -> list[set[int]]:
        """
        Returns a list of all the valid qudit groups in the coupling map. 

        Args:
            None 

        Returns:
            qudit_groups (Sequence[Sequence[int]]): A list of all groups of 
                physically connected qudits that are < block_size away from each
                other.

        Notes:
            Does a breadth first search on all pairs of qudits, keeps paths
            that have length equal to block_size. Note that the coupling map
            is assumed to be undirected.

        """
        # Create an adjaceny dict
        adj_dict = {k:[] for k in range(self.num_qudits)}
        for edge in self.machine.coupling_graph:
            adj_dict[edge[0]].append(edge[1])
            adj_dict[edge[1]].append(edge[0])

        found_paths = set([])
        # For each qudit
        for vertex in range(self.num_qudits):
            path = set()
            # Get every vald set containing s with cardinality == block_size
            self._qudit_group_search(
                adj_dict = adj_dict, 
                all_paths = found_paths, 
                path = path, 
                vertex = vertex, 
                limit = self.block_size
            )

        # Return the set as a list of paths/qudit groups
        list_of_paths = []
        for path in found_paths:
            list_of_paths.append(path)
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
        if vertex in path:
            return
        curr_path = path.copy()
        curr_path.add(vertex)
        if len(curr_path) == limit:
            all_paths.add(frozenset(curr_path))
        else:
            frontier = []
            for node in curr_path:
                frontier.extend(adj_dict[node])
            for neighbor in frontier:
                if neighbor not in curr_path:
                    self._qudit_group_search(
                        adj_dict = adj_dict, 
                        all_paths = all_paths, 
                        path = curr_path, 
                        vertex = neighbor, 
                        limit = limit
                    )

    class BlockInfo:
        """
        The BlockInfo Class.

        A BlockInfo object contains the description of a synthesizable block.

        Notes: For each qudit, the block_start is inclusive and the block_end 
            is exclusive, meaning the block_end cycle should not be included 
            in the synthesizable block.
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
            self.block_start = {k:0 for k in qudits} if qudits is not None \
                else {}
            self.block_end   = {k:0 for k in qudits} if qudits is not None \
                else {}
            self.score = score

    def _set_run_parameters(
        self, 
        circuit: Circuit, 
        data: dict[str, Any]
    ) -> None:
        """
        Set up the SimplePartitioner variables for the current run call.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str, Any]): Additional data for the run.
        """
        # num_qudits is the number of physical qubits in the topology
        self.num_qudits = data["num_qudits"] if "num_qudits" in data \
            else self.num_qudits
        # num_verts is the number of qubits used in the algorithm/circuit
        self.num_verts = circuit.get_size()
        # Cannot have more algorithmic qubits than physical qubits
        if self.num_qudits < self.num_verts:
            raise ValueError(
                'Expected num_qudits >= circuit.get_size(), '
                'got %d (num_qudits) and %d' %(self.num_qudits, self.num_verts)
            )

        # If a coupling map is provided in the data dict, it will be used. 
        # Otherwise the coupling graph assigned to the MachineModel will be
        # used. 
        self.coupling_graph = data["coupling_graph"] if "coupling_graph" in \
            data else self.machine.coupling_graph

        self.block_size = data["block_size"] if "block_size" in data \
            else self.block_size
        if self.block_size < 2 or self.block_size > self.num_qudits:
            raise ValueError(
                'Expected  2 <= block_size <= num_qudits, '
                'got %d' %(self.block_size)
            )

        # Change scores for multi and single qudit gates if desired
        if "multi_gate_score" in data:
            self.multi_gate_score = data["multi_gate_score"]
        if "single_gate_score" in data:
            self.single_gate_score = data["single_gate_score"]

        # Get the set of all vertices used in the algorithm
        #self.used_verts = get_used_qudit_set(circuit)

    def num_ops_left(
        self,
        circuit : Circuit,
        qudit : int,
        cycle : int
    ) -> int:
        """
        Return the number of operations on a qudit at and beyond the cycle
        specified by the point given.

        Args:
            circuit (Circuit): Circuit to be analyzed.

            qudit (int): Qudit on which to find remaining operations.

            cycle (int): Cycle at and beyond which to count operations.

        Returns:
            int: The number of operations at and beyond the given point.
        """
        qudit_iter = circuit.operations_on_qudit_with_points(qudit)
        num_ops = 0
        for circ_point, circ_op in qudit_iter:
            if circ_point.cycle >= cycle:
                num_ops += 1
        return num_ops


    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates. 

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.

        Raises:
            ValueError: If the number of qudits in the circuit is larger than
                the number of qudits in the coupling map.

            ValueError: If the blocksize is too big or too small.
        """
        self._set_run_parameters(circuit, data)

        # Find all paths between any two used vertices that is less than the
        # synthesizable blocksize
        self.qudit_groups = self.get_qudit_groups()

        # Do while there are still gates to partition
        # NOTE: This assumes circuit and topology qudit numbers are equal
        sched_depth = [0 for i in range(self.num_qudits)]
        while not all(
            [self.num_ops_left(circuit, sched_depth[p], p) for p 
                in range(self.num_qudits)]
        ):
            # Find the scores of the qudit groups.
            best_block = self.BlockInfo()
            for q_group in self.qudit_groups:
                # Examine each qudit in the group
                curr_block = self.BlockInfo(q_group)
                # Earliest cycle not partitioned is the scheduled depth
                for qudit in q_group:
                    curr_block.block_start[qudit] = sched_depth[qudit]

                # Find the earliest cycle in the group in which each q is 
                # involved with an "outsider" qudit 
                insider_qudits = q_group
                # TODO: CircuitIterator is costly to use, find a way to iterate
                #   through a subcircuit in order.
                circ_iter = Circuit.SubCircuitIterator(
                    circuit = circuit._circuit, 
                    subset = q_group,
                    and_points = True)
                # Skip operations that are before the block_start or do not
                # involve qudits in the insider_qudits
                for point, op in circ_iter:
                    # Check if we have finished partitioning the current block
                    if len(insider_qudits) == 0:
                        break
                    # Do not count outside block limits
                    elif point.cycle < sched_depth[point.qudit]:
                        continue

                    # Check that op doesn't interact with outsider qudits.
                    # If a qudit interacts with an outsider, mark the end
                    # of the block for it and add it to the outsiders.
                    if any(other_qudit not in insider_qudits for other_qudit 
                        in op.location):
                        curr_block.block_end[point.qudit] = max(point.cycle-1,0)
                        insider_qudits -= set([point.qudit])
                    # Else add to the score of the current block, update the
                    # end of the block for all still valid insider qudits.
                    else:
                        for insider in insider_qudits:
                            curr_block.block_end[insider] = point.cycle
                        if len(op.location) >= 2:
                            curr_block.score += self.multi_gate_score
                        else:
                            curr_block.score += self.single_gate_score

                # Update the best_block if the current block scores higher
                if curr_block.score > best_block.score:
                    best_block = curr_block

            # Replace the highest scoring block with a CircuitGate.
            qudits = best_block.block_start.keys()
            points_in_block = []
            circ_iter = circuit.SubCircuitIterator(
                circuit = circuit._circuit,
                subset = qudits,
                and_points = True
            )
            for point, op in circ_iter:
                if point.cycle >= best_block.block_start[point.qudit] and \
                    point.cycle <= best_block.block_end[point.qudit] and \
                        all(other_qs in qudits for other_qs in op.location):
                    points_in_block.append(point)

            circuit.fold(points_in_block)

            # Update the scheduled depth list
            for q in qudits:
                sched_depth[q] += 1
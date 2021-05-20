"""This module defines the GreedyPartitioner pass."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit

# TODO:
#   Layout should be a separate pass from partitioning. The partitioner may
#   need to be changed so that it can accept a layout assignment, but by
#   default assumes the numberings in the algorithm and topology are equal.


class GreedyPartitioner(BasePass):
    # Class variables
    used_qudits = set()     # set[int]
    qudit_groups = []       # list[set[int]]

    def __init__(
        self,
        block_size: int = 3,
    ) -> None:
        """
        Constructor for a GreedyPartitioner.

        Args:
            block_size (int): Size of synthesizable partition blocks.
        """
        self.block_size = block_size

        # Default scoring method
        self.multi_gate_score = 1000
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
            qudit_groups (list[set[int]]): A list of all groups of
                physically connected qudits that are < block_size away from each
                other.

        Notes:
            Does a breadth first search on all pairs of qudits, keeps paths
            that have length equal to block_size. Note that the coupling map
            is assumed to be undirected.
        """
        # Create an adjaceny dict
        adj_dict = {k: [] for k in range(self.num_verts)}
        for edge in self.coupling_graph:
            if edge[0] < self.num_verts and edge[1] < self.num_verts:
                adj_dict[edge[0]].append(edge[1])
                adj_dict[edge[1]].append(edge[0])

        found_paths = set()
        # For each qudit
        for vertex in range(self.num_verts):
            path = set()
            # Get every valid set containing vertex with cardinality ==
            # block_size
            self._qudit_group_search(
                adj_dict=adj_dict,
                all_paths=found_paths,
                path=path,
                vertex=vertex,
                limit=self.block_size,
            )

        # Return the set as a list of paths/qudit groups
        list_of_paths = []
        for path in found_paths:
            list_of_paths.append(path)
        return list_of_paths

    def _qudit_group_search(
        self,
        adj_dict: dict[int, int],
        all_paths: set[frozenset[int]],
        path: set[int],
        vertex: int,
        limit: int,
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
                        adj_dict=adj_dict,
                        all_paths=all_paths,
                        path=curr_path,
                        vertex=neighbor,
                        limit=limit,
                    )

    def _set_run_parameters(
        self,
        circuit: Circuit,
        data: dict[str, Any],
    ) -> None:
        """
        Set up the GreedyPartitioner variables for the current run call.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str, Any]): Additional data for the run.
        """
        # num_verts is the number of qudits used in the algorithm/circuit
        self.num_verts = circuit.get_size()

        # coupling map is the adjacency graph of the circuit
        self.coupling_graph = circuit.get_coupling_graph()

        self.block_size = data['block_size'] if 'block_size' in data \
            else self.block_size
        if self.block_size < 2 or self.block_size > self.num_verts:
            raise ValueError(
                'Expected  2 <= block_size <= num_verts, '
                'got %d' % (self.block_size),
            )

        # Change scores for multi and single qudit gates if desired
        if 'multi_gate_score' in data:
            self.multi_gate_score = data['multi_gate_score']
        if 'single_gate_score' in data:
            self.single_gate_score = data['single_gate_score']

        # Get the set of all vertices used in the algorithm
        #self.used_verts = get_used_qudit_set(circuit)

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
        self.qudit_groups = [
            list(q_group)
            for q_group in self.get_qudit_groups()
        ]

        num_cycles = circuit.get_num_cycles()
        num_qudits_groups = len(self.qudit_groups)

        op_cycles = [[[0] * self.block_size for q_group in self.qudit_groups]
                     for cycle in range(num_cycles)]
        for point, op in circuit.operations_with_points():
            cycle = point.cycle
            if len(op.location) > 1:
                for q_group_index, q_group in enumerate(self.qudit_groups):
                    if all([qudit in q_group for qudit in op.location]):
                        for qudit in op.location:
                            op_cycles[cycle][q_group_index][
                                q_group.index(
                                qudit,
                                )
                            ] = self.multi_gate_score
                    else:
                        for qudit in op.location:
                            if qudit in q_group:
                                op_cycles[cycle][q_group_index][
                                    q_group.index(
                                    qudit,
                                    )
                                ] = -1
            else:
                qudit = point.qudit
                for q_group_index, q_group in enumerate(self.qudit_groups):
                    if qudit in q_group:
                        op_cycles[cycle][q_group_index][
                            q_group.index(
                            qudit,
                            )
                        ] = self.single_gate_score

        max_blocks = []
        for q_group_index in range(num_qudits_groups):
            block_start = 0
            block_ends = [0] * self.block_size
            score = 0
            for cycle in range(num_cycles):
                if cycle:
                    for qudit in range(self.block_size):
                        if op_cycles[cycle - 1][q_group_index][qudit] == - \
                                1 and op_cycles[cycle][q_group_index][qudit] != -1:
                            max_blocks.append(
                                [score, block_start, block_ends, q_group_index],
                            )
                            score = 0
                            block_start = cycle
                            block_ends = [cycle + 1] * self.block_size
                            break
                for qudit in range(self.block_size):
                    if op_cycles[cycle][q_group_index][qudit] != -1:
                        block_ends[qudit] = cycle + 1
                        score += op_cycles[cycle][q_group_index][qudit]
            max_blocks.append([score, block_start, block_ends, q_group_index])

        block_id = -1
        new_blocks = []
        max_blocks.sort()
        remaining_assignments = self.num_verts * num_cycles
        block_map = [[-1] * self.num_verts for cycle in range(num_cycles)]
        while remaining_assignments:

            perform_assign = False
            if len(max_blocks) == 1:
                perform_assign = True
            else:
                block_start = max_blocks[-1][1]
                block_ends = max_blocks[-1][2]
                q_group_index = max_blocks[-1][3]
                score = 0
                for cycle in range(block_start, max(block_ends)):
                    for qudit in range(self.block_size):
                        if cycle < block_ends[qudit] and block_map[cycle][self.qudit_groups[q_group_index][qudit]] == -1:
                            score += 1
                if score < max_blocks[-2][0]:
                    max_blocks[-1][0] = score
                    max_blocks.sort()
                else:
                    perform_assign = True

            if perform_assign:
                block_id += 1
                block_start = max_blocks[-1][1]
                block_ends = max_blocks[-1][2]
                q_group_index = max_blocks[-1][3]
                prev_status = None
                for cycle in range(block_start, max(block_ends)):
                    status = [
                        block_map[cycle][self.qudit_groups[q_group_index][qudit]]
                        for qudit in range(self.block_size)
                    ]
                    if prev_status and len(prev_status) <= len(
                            status,
                    ) and status != prev_status:
                        block_id += 1
                    for qudit in range(self.block_size):
                        if cycle < block_ends[qudit] and block_map[cycle][self.qudit_groups[q_group_index][qudit]] == -1:
                            block_map[cycle][
                                self.qudit_groups[q_group_index]
                                [qudit]
                            ] = block_id
                            remaining_assignments -= 1
                    prev_status = status
                del max_blocks[-1]

        for cycle in range(num_cycles):
            if not cycle or block_map[cycle] == block_map[cycle - 1]:
                continue
            indices = [{}, {}]
            for i in range(2):
                for qudit in range(self.num_verts):
                    block = block_map[cycle - i][qudit]
                    if block not in indices[i]:
                        indices[i][block] = []
                    indices[i][block].append(qudit)
            for prev_blocks, prev_qudits in indices[1].items():
                for current_qudits in indices[0].values():
                    if all([qudit in prev_qudits for qudit in current_qudits]):
                        for qudit in current_qudits:
                            block_map[cycle][qudit] = prev_blocks
            prev_cycle = cycle

        blocks = {}
        for cycle in range(num_cycles):
            for qudit in range(self.num_verts):
                if block_map[cycle][qudit] not in blocks:
                    blocks[block_map[cycle][qudit]] = {}
                    blocks[block_map[cycle][qudit]][-1] = cycle
                blocks[block_map[cycle][qudit]][qudit] = cycle

        block_order = []
        for block in blocks.values():
            block_order.append([block, block[-1]])
        block_order.sort(reverse=True, key=lambda x: x[1])

        for block, start_cycle in block_order:
            points_in_block = []
            for point, op in circuit.operations_with_points():
                cycle = point.cycle
                qudit = point.qudit
                if qudit in block and cycle >= start_cycle and cycle <= block[qudit]:
                    points_in_block.append(point)

            circuit.fold(points_in_block)

"""This module implements the SubtopologySelectionPass pass."""
from __future__ import annotations

import copy
import itertools as it
from typing import Any
from typing import Iterable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.ir import Circuit
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.graph import CouplingGraphLike
from bqskit.utils.typing import is_integer


def powerset(iterable: Iterable[Any]) -> Iterable[Any]:
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    all_combos = (it.combinations(s, r) for r in range(len(s) + 1))
    return it.chain.from_iterable(all_combos)


def all_coupling_graphs_of_size(n: int) -> list[CouplingGraph]:
    """Calculates all valid coupling graphs with `n` qudits."""
    if not is_integer(n):
        raise TypeError(f'Expected integer for n, got {type(n)}.')

    if n <= 1:
        raise ValueError(f'Smallest valid graph is 2, got {n}.')

    graphs: list[list[list[tuple[int, int]]]] = [[[], [(0, 1)]]]

    while len(graphs) + 1 < n:
        i = len(graphs) + 1
        extended_graph_list = []
        for graph in graphs[-1]:
            for nodes in powerset(range(i)):
                edges = copy.deepcopy(graph)

                for node in nodes:
                    edges.append((node, i))

                extended_graph_list.append(edges)
        graphs.append(extended_graph_list)

    to_ret = []
    for g in graphs[-1]:
        cg = CouplingGraph(g, n)
        if cg.is_fully_connected():
            to_ret.append(cg)

    return to_ret


class GraphDAGNode():
    """A node in the GraphDAG."""

    def __init__(self, graph_id: int):
        self.indices: list[int] = [graph_id]
        self.predecessors: list[GraphDAGNode] = []
        self.successors: list[GraphDAGNode] = []

    def peek_index(self) -> int:
        """Provide index of this kind of graph in list."""
        return self.indices[0]


class GraphDAG():
    """
    DAG for organizing the embeddability relations of graphs.

    The DAG is organized so that a node is guaranteed to be embedded within all
    of its successor nodes.
    """

    def __init__(self, graph_list: list[CouplingGraphLike]):
        self.roots: list[GraphDAGNode] = []
        self.leafs: list[GraphDAGNode] = []
        self.graph_list = graph_list
        if not self.graph_list[0]._edges == set():
            self.graph_list.insert(0, CouplingGraph([]))
        self._create_DAG()
        self._insert_node(0)

    def _create_DAG(self) -> None:
        for graph_id in range(len(self.graph_list)):
            self._insert_node(graph_id)

    def _forward_path_exists(self, node_a: Any, node_b: Any) -> bool:
        """Return True if a forward path exists from node_a to node_b."""
        if node_a is node_b or node_b in node_a.successors:
            return True
        to_check = node_a.successors.copy()
        for node in to_check:
            if node_b in node.successors:
                return True
            else:
                to_check.extend(node.successors)
        return False

    def _insert_node(self, graph_id: int) -> None:
        """
        Insert a new node corresponding to the graph `graph_list[graph_id]` into
        the DAG.

        Append it to the "least embeddable" graph that is still embedded within
        it.
        """
        node = GraphDAGNode(graph_id)
        if len(self.roots) == 0:
            self.roots.append(node)
            self.leafs.append(node)
            return

        graph = self.graph_list[graph_id]
        nodes_to_check = self.leafs.copy()

        # Check for node to be predessor
        for other_node in nodes_to_check:
            other_graph = self.graph_list[
                other_node.peek_index()
            ]

            if other_graph.is_embedded_in(graph):
                # Handle isomorphic case
                if graph.is_embedded_in(other_graph):
                    other_node.indices.append(graph_id)
                    return
                if not self._forward_path_exists(other_node, node):
                    other_node.successors.append(node)
                    node.predecessors.append(other_node)

                # Check if we were at a leaf
                if other_node in self.leafs:
                    self.leafs.remove(other_node)
                if node not in self.leafs:
                    self.leafs.append(node)
            else:
                # Only check predecessors if node is not embedded in
                # leaf and there is not already a direct forward path
                # to the node
                nodes_to_check += other_node.predecessors

    def get_embedded_indices(
        self,
        graphs: list[CouplingGraph] | CouplingGraph,
    ) -> list[int]:
        """Given a list of graphs, return the indices of subgraphs in the
        `self.graph_list` list that are embedded in a subgraph in `graphs`."""
        embedded_indices: set[int] = set()
        if type(graphs) is CouplingGraph:
            graphs = [graphs]

        for graph in graphs:
            # Check least embeddable graphs first
            to_check: list[GraphDAGNode] = self.leafs.copy()
            for node in to_check:
                subgraph = self.graph_list[node.peek_index()]
                if not subgraph.is_embedded_in(graph):
                    # If not embedded, check predecessors
                    to_check.extend(node.predecessors)
                else:
                    # If embedded, add node's indices and all predecessors
                    new_id = node.peek_index()
                    if new_id not in embedded_indices:
                        embedded_indices = embedded_indices.union(
                            set(node.indices),
                        )
                        preds = node.predecessors
                        for p in preds:
                            if p.peek_index() not in embedded_indices:
                                embedded_indices = embedded_indices.union(
                                    set(p.indices),
                                )
                                preds.extend(p.predecessors)
        return sorted(list(embedded_indices))


def filter_compatible_subgraphs(
    candidate_subgraphs: list[CouplingGraphLike],
    machine: MachineModel,
    blocksize: int | None = None,
) -> list[CouplingGraph]:
    """
    Filter the candidate subgraphs, returning the ones appearing in `machine`.

    Given a list of candidate synthesis subgraphs and a MachineModel with a
    valid defined CouplingGraph, return a list of candidate subgraphs that are
    embedded within the machine.

    Arguments:
        candidate_subgraphs (list[CouplingGraph]): A list containing subgraphs
            that should be checked for embeddability within some MachineModel.

        machine (MachineModel): A MachineModel with a valid and specified
            CouplingGraph.

        blocksize (int | None): Maximum desired subgraph size. If unspecified,
            the entire CouplingGraph will be searched. It is highly recommended
            that the blocksize be provided for performance reasons.
            (Default: None)

    Returns:
        (list[CouplingGraph]): A list of coupling graphs that are both in the
            `candidate_subgraphs` list and the `machine`'s CouplingGraph.

    Raises:
        ValueError: If the CouplingGraph of the `machine` variable is either
            invalid or unspecified.
    """
    graph = machine.coupling_graph
    if graph is None or not graph.is_valid_coupling_graph(graph):
        raise ValueError(
            'The CouplingGraph specified by MachineModel `machine` '
            'is unspecified or not fully connected.',
        )
    if blocksize is None:
        blocksize = graph.num_qudits

    # Sort graphs by number of edge
    locations = graph.get_subgraphs_of_size(blocksize)
    induced_subgraphs: list[CouplingGraph] = sorted(
        (
            CouplingGraph(graph.get_induced_subgraph(l)).relabel_subgraph()
            for l in locations
        ),
        key=lambda x: -len(x),
    )
    candidates = sorted(candidate_subgraphs, key=lambda x: len(x))
    dag = GraphDAG(candidates)
    indices_of_interest = dag.get_embedded_indices(induced_subgraphs)

    to_return = [
        candidates[i] for i in range(len(candidates))
        if i in indices_of_interest
    ]
    # Do not return empty set used as root
    if to_return[0]._edges == set():
        return to_return[1:]
    return to_return


class SubtopologySelectionPass(BasePass):
    """Pass that selects necessary subtopologies from the model."""

    key = ForEachBlockPass.pass_down_key_prefix + 'sub_topologies'

    def __init__(self, block_size: int) -> None:
        """
        Construct a SubtopologySelectionPass.

        Args:
            block_size (int): The max block size to select subtopologies for.

        Raises:
            ValueError: If block_size is <= 1.
        """
        if not is_integer(block_size):
            raise TypeError(f'Expected integer, got {type(block_size)}.')

        if block_size <= 1:
            raise ValueError(f'Expected integer > 1, got {block_size}.')

        self.block_size = block_size

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        model = data.model
        n = model.num_qudits
        cg = model.coupling_graph
        all_to_all = len(cg._edges) == n * (n - 1) / 2

        tops = {}
        for i in range(2, self.block_size + 1):
            if all_to_all:
                tops[i] = [CouplingGraph.all_to_all(i)]
            else:
                all_tops = all_coupling_graphs_of_size(i)
                tops[i] = filter_compatible_subgraphs(all_tops, model, i)

        data[self.key] = tops

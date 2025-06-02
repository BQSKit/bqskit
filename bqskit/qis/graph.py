"""This module implements the CouplingGraph class."""
from __future__ import annotations

import copy
import itertools as it
import logging
import warnings
from random import shuffle
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

from bqskit.ir.location import CircuitLocation
from bqskit.ir.location import CircuitLocationLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable, is_mapping, is_real_number

_logger = logging.getLogger(__name__)


class CouplingGraph(Collection[Tuple[int, int]]):
    """A graph representing connections in a qudit topology."""

    def __init__(
        self,
        graph: CouplingGraphLike,
        num_qudits: int | None = None,
        remote_edges: Iterable[tuple[int, int]] = [],
        default_weight: float = 1.0,
        default_remote_weight: float = 100.0,
        edge_weights_overrides: Mapping[tuple[int, int], float] = {},
    ) -> None:
        """
        Construct a new CouplingGraph.

        Args:
            graph (CouplingGraphLike): The undirected graph edges.

            num_qudits (int | None): The number of qudits in the graph. If
                None, the number of qudits is inferred from the maximum seen
                in the edge list. (Default: None)

            remote_edges (Iterable[tuple[int, int]]): The edges that cross
                QPU chip boundaries. Distributed QPUs will have remote links
                connect them. Notes, remote edges must specified both in
                `graph` and here. (Default: [])

            default_weight (float): The default weight of an edge in the
                graph. (Default: 1.0)

            default_remote_weight (float): The default weight of a remote
                edge in the graph. (Default: 100.0)

            edge_weights_overrides (Mapping[tuple[int, int], float]): A mapping
                of edges to their weights. These override the defaults on
                a case-by-case basis. (Default: {})

        Raises:
            ValueError: If `num_qudits` is too small for the edges in `graph`.

            ValueError: If `num_qudits` is less than zero.

            ValueError: If any edge in `remote_edges` is not in `graph`.

            ValueError: If any edge in `edge_weights_overrides` is not in
                `graph`.
        """
        if not CouplingGraph.is_valid_coupling_graph(graph):
            raise TypeError('Invalid coupling graph.')

        if num_qudits is not None and not is_integer(num_qudits):
            raise TypeError(
                'Expected integer for num_qudits,'
                f' got {type(num_qudits)}',
            )

        if num_qudits is not None and num_qudits < 0:
            raise ValueError(
                'Expected nonnegative num_qudits,'
                f' got {num_qudits}.',
            )

        if not CouplingGraph.is_valid_coupling_graph(remote_edges):
            raise TypeError('Invalid remote links.')

        if any(edge not in graph for edge in remote_edges):
            invalids = [e for e in remote_edges if e not in graph]
            raise ValueError(
                f'Remote links {invalids} not in graph.'
                ' All remote links must also be specified in the graph input.',
            )

        if not is_real_number(default_weight):
            raise TypeError(
                'Expected integer for default_weight,'
                f' got {type(default_weight)}',
            )

        if not is_real_number(default_remote_weight):
            raise TypeError(
                'Expected integer for default_remote_weight,'
                f' got {type(default_remote_weight)}',
            )

        if not is_mapping(edge_weights_overrides):
            raise TypeError(
                'Expected mapping for edge_weights_overrides,'
                f' got {type(edge_weights_overrides)}',
            )

        if any(
            not is_real_number(v)
            for v in edge_weights_overrides.values()
        ):
            invalids = [
                v for v in edge_weights_overrides.values()
                if not is_real_number(v)
            ]
            raise TypeError(
                'Expected integer values for edge_weights_overrides,'
                f' got non-integer values: {invalids}.',
            )

        if any(edge not in graph for edge in edge_weights_overrides):
            invalids = [
                e for e in edge_weights_overrides
                if e not in graph
            ]
            raise ValueError(
                f'Edges {invalids} from edge_weights_overrides are not in '
                'the graph. All edge_weights_overrides must also be '
                'specified in the graph input.',
            )

        if isinstance(graph, CouplingGraph):
            self.num_qudits: int = graph.num_qudits
            self._edges: set[tuple[int, int]] = graph._edges
            self._remote_edges: set[tuple[int, int]] = graph._remote_edges
            self._adj: list[set[int]] = graph._adj
            self._mat: list[list[float]] = graph._mat
            self.default_weight: float = graph.default_weight
            self.default_remote_weight: float = graph.default_remote_weight
            return

        calc_num_qudits = 0
        for q1, q2 in graph:
            calc_num_qudits = max(calc_num_qudits, max(q1, q2))
        calc_num_qudits += 1

        if num_qudits is not None and calc_num_qudits > num_qudits:
            raise ValueError(
                'Edges between invalid qudits or num_qudits too small.',
            )

        self.num_qudits = calc_num_qudits if num_qudits is None else num_qudits
        self._edges = {g if g[0] <= g[1] else (g[1], g[0]) for g in graph}
        self._remote_edges = {
            e if e[0] <= e[1] else (e[1], e[0])
            for e in remote_edges
        }
        self.default_weight = default_weight
        self.default_remote_weight = default_remote_weight

        self._adj = [set() for _ in range(self.num_qudits)]
        for q1, q2 in self._edges:
            self._adj[q1].add(q2)
            self._adj[q2].add(q1)

        self._mat = [
            [np.inf for _ in range(self.num_qudits)]
            for _ in range(self.num_qudits)
        ]
        for q1, q2 in self._edges:
            self._mat[q1][q2] = default_weight
            self._mat[q2][q1] = default_weight

        for q1, q2 in self._remote_edges:
            self._mat[q1][q2] = default_remote_weight
            self._mat[q2][q1] = default_remote_weight

        for (q1, q2), weight in edge_weights_overrides.items():
            self._mat[q1][q2] = weight
            self._mat[q2][q1] = weight

    def get_qpu_to_qudit_map(self) -> list[list[int]]:
        """Return a mapping of QPU indices to qudit indices."""
        if not hasattr(self, '_qpu_to_qudit'):
            seen = set()
            self._qpu_to_qudit = []
            for qudit in range(self.num_qudits):
                if qudit in seen:
                    continue
                qpu = []
                frontier = {qudit}
                while len(frontier) > 0:
                    node = frontier.pop()
                    qpu.append(node)
                    seen.add(node)
                    for neighbor in self._adj[node]:
                        if (node, neighbor) in self._remote_edges:
                            continue
                        if (neighbor, node) in self._remote_edges:
                            continue
                        if neighbor not in seen:
                            frontier.add(neighbor)
                self._qpu_to_qudit.append(qpu)
        return self._qpu_to_qudit

    def is_distributed(self) -> bool:
        """Return true if the graph represents multiple connected QPUs."""
        return len(self._remote_edges) > 0

    def qpu_count(self) -> int:
        """Return the number of connected QPUs."""
        return len(self.get_qpu_to_qudit_map())

    def get_individual_qpu_graphs(self) -> list[CouplingGraph]:
        """Return a list of individual QPU graphs."""
        if not self.is_distributed():
            return [self]

        qpu_to_qudit = self.get_qpu_to_qudit_map()
        return [self.get_subgraph(qpu) for qpu in qpu_to_qudit]

    def get_qudit_to_qpu_map(self) -> list[int]:
        """Return a mapping of qudit indices to QPU indices."""
        qpu_to_qudit = self.get_qpu_to_qudit_map()
        qudit_to_qpu = {}
        for qpu, qudits in enumerate(qpu_to_qudit):
            for qudit in qudits:
                qudit_to_qpu[qudit] = qpu
        return list(qudit_to_qpu.values())

    def get_qpu_connectivity(self) -> list[set[int]]:
        """Return the adjacency list of the QPUs."""
        qpu_to_qudit = self.get_qpu_to_qudit_map()
        qudit_to_qpu = self.get_qudit_to_qpu_map()
        qpu_adj: list[set[int]] = [set() for _ in range(len(qpu_to_qudit))]
        for q1, q2 in self._remote_edges:
            qpu1 = qudit_to_qpu[q1]
            qpu2 = qudit_to_qpu[q2]
            qpu_adj[qpu1].add(qpu2)
            qpu_adj[qpu2].add(qpu1)
        return qpu_adj

    def is_fully_connected(self) -> bool:
        """Return true if the graph is fully connected."""
        frontier: set[int] = {0}
        qudits_seen: set[int] = set()

        while len(frontier) > 0:
            expanded_qudits = set()

            for qudit in frontier:
                expanded_qudits.add(qudit)
                expanded_qudits.update(self._adj[qudit])

            frontier = expanded_qudits - qudits_seen
            qudits_seen.update(expanded_qudits)

            if len(qudits_seen) == self.num_qudits:
                return True

        return False

    def is_linear(self) -> bool:
        """Return true if the graph is linearly connected."""
        if self.num_qudits < 2:
            return False

        num_deg_1 = 0
        for node_neighbors in self._adj:
            if len(node_neighbors) == 1:
                num_deg_1 += 1

            elif len(node_neighbors) == 0:
                return False

            elif len(node_neighbors) > 2:
                return False

        if num_deg_1 != 2:
            return False

        return True

    def get_neighbors_of(self, qudit: int) -> list[int]:
        """Return the qudits adjacent to `qudit`."""
        return list(self._adj[qudit])

    def __contains__(self, __o: object) -> bool:
        return self._edges.__contains__(__o)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, CouplingGraph):
            return False

        if self.num_qudits != __o.num_qudits:
            return False

        if self._mat != __o._mat:
            return False

        return True

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return self._edges.__iter__()

    def __hash__(self) -> int:
        return hash((self.num_qudits, tuple(self._edges)))

    def __len__(self) -> int:
        return self._edges.__len__()

    def __str__(self) -> str:
        return 'CouplingGraph(' + self._edges.__str__() + ')'

    def __repr__(self) -> str:
        return self._edges.__repr__()

    def get_qudit_degrees(self) -> list[int]:
        return [len(l) for l in self._adj]

    def all_pairs_shortest_path(self) -> list[list[float]]:
        """
        Calculate all pairs shortest path matrix using Floyd-Warshall.

        Returns:
            D (list[list[float]]): D[i][j] is the length of the shortest
                path from i to j.
        """
        D = copy.deepcopy(self._mat)
        for k in range(self.num_qudits):
            for i in range(self.num_qudits):
                for j in range(self.num_qudits):
                    D[i][j] = min(D[i][j], D[i][k] + D[k][j])
        return D

    def get_shortest_path_tree(self, source: int) -> list[tuple[int, ...]]:
        """Return shortest path from `source` to every node in `self`."""
        # Dijkstra's algorithm to build shortest-path tree
        unvisited_qudits = set(range(self.num_qudits))
        distances = {i: np.inf for i in range(self.num_qudits)}
        paths: list[tuple[int, ...]] = [tuple() for i in range(self.num_qudits)]
        distances[source] = 0
        paths[source] = (source,)

        while len(unvisited_qudits) > 0:
            # Pick next unvisited qudit with shortest distance
            unvisited_distances = [
                (x, y) for x, y in distances.items() if x in unvisited_qudits
            ]
            unvisited_distances.sort(key=lambda x: x[1])
            current_qudit = unvisited_distances[0][0]

            if distances[current_qudit] == np.inf:
                raise RuntimeError(f'No path found to qudit: {current_qudit}.')

            neighbors = self.get_neighbors_of(current_qudit)
            unvisted_neighbors = unvisited_qudits.intersection(neighbors)

            for other_qudit in unvisted_neighbors:
                if distances[current_qudit] + 1 < distances[other_qudit]:
                    distances[other_qudit] = distances[current_qudit] + 1
                    paths[other_qudit] = paths[current_qudit] + (other_qudit,)

            unvisited_qudits.remove(current_qudit)

        return [paths[i] for i in range(self.num_qudits)]

    def get_subgraph(
        self,
        location: CircuitLocationLike,
        renumbering: dict[int, int] | None = None,
    ) -> CouplingGraph:
        """
        Returns a sub-coupling-graph induced by the qudits in `location`.

        The method constructs a new CouplingGraph that contains only the qudits
        specified in `location`, and only the edges between them that exist
        in the original coupling graph. By default, the qudits are renumbered
        to consecutive integers starting from 0 in the order they appear in `location`.

        Parameters
        ----------
        location : CircuitLocationLike
            A list or iterable of qudit indices specifying which subset of the
            original graph to extract.

        renumbering : dict[int, int], optional
            A dictionary mapping each qudit index in `location` to a new index
            in the returned subgraph. If not provided, the qudits are
            automatically renumbered to consecutive integers starting from 0.

        Returns
        -------
        CouplingGraph
            A new CouplingGraph object representing the induced subgraph, where
            the nodes and edges correspond to those in the original graph,
            but remapped according to `renumbering` if provided.

        """
        if not CircuitLocation.is_location(location, self.num_qudits):
            raise TypeError('Invalid location.')

        location = CircuitLocation(location)
        if renumbering is None:
            renumbering = {q: i for i, q in enumerate(location)}

        subgraph = []
        location_set = {loc for loc in location}
        for q_i in location:
            for q_i_neighbor in location_set.intersection(self._adj[q_i]):
                subgraph.append((renumbering[q_i], renumbering[q_i_neighbor]))
        return CouplingGraph(subgraph, len(location))

    def get_subgraphs_of_size(self, size: int) -> list[CircuitLocation]:
        """
        Find all sets of indices that form connected subgraphs on their own.

        Each location describes a valid spot for a `size`-sized gate,
        so the number of qudit_indices in each location is `size`.
        A location is only included if each pair of qudits is directly
        connected or connected through other qudits in the location.

        Args:
            size (int): The size of each location in the final list.

        Returns:
            list[CircuitLocation]: The locations compliant with the machine.

        Raises:
            ValueError: If `size` is nonpositive or too large.

        Notes:
            Does a breadth first search on all pairs of qudits, keeps paths
            that have length equal to size. Note that the coupling map
            is assumed to be undirected.
        """
        if not is_integer(size):
            raise TypeError(
                f'Expected integer for block_size, got {type(size)}',
            )

        if size > self.num_qudits:
            raise ValueError(
                'The block_size is too large; '
                f'graph size is {self.num_qudits}, got {size}.',
            )

        if size <= 0:
            raise ValueError(f'Nonpositive size; got {size}.')

        locations: set[CircuitLocation] = set()

        for qudit in range(self.num_qudits):
            # Get every valid set containing `qudit` with size == size
            self._location_search(locations, set(), qudit, size)

        return list(locations)

    def _location_search(
        self,
        locations: set[CircuitLocation],
        path: set[int],
        vertex: int,
        limit: int,
    ) -> None:
        """
        Add paths with length equal to limit to the `locations` set.

        Args:
            locations (set[CircuitLocation]): A list that contains all paths
                found so far of length equal to `limit`.

            path (set[int]): The qudit vertices currently included in
                the path.

            vertex (int): The vertex in the graph currently being examined.

            limit (int): The desired length of paths in the `locations`
                list.
        """
        if vertex in path:
            return

        curr_path = path.copy()
        curr_path.add(vertex)

        if len(curr_path) == limit:
            locations.add(CircuitLocation(list(curr_path)))
            return

        frontier: set[int] = {
            qudit
            for node in curr_path
            for qudit in self._adj[node]
            if qudit not in curr_path
        }

        for neighbor in frontier:
            self._location_search(locations, curr_path, neighbor, limit)

    @staticmethod
    def is_valid_coupling_graph(
        coupling_graph: Any,
        num_qudits: int | None = None,
    ) -> TypeGuard[CouplingGraphLike]:
        """
        Return true if the coupling graph is valid.

        Args:
            coupling_graph (Any): The coupling graph to check.

            num_qudits (int | None): The total number of qudits. All qudits
                should be less than this. If None, don't check.

        Returns:
            bool: Valid or not
        """

        if not is_iterable(coupling_graph):
            _logger.debug('Coupling graph is not iterable.')
            return False

        if len(list(coupling_graph)) == 0:
            return True

        if not all(isinstance(pair, tuple) for pair in coupling_graph):
            _logger.debug('Coupling graph is not a sequence of tuples.')
            return False

        if not all([len(pair) == 2 for pair in coupling_graph]):
            _logger.debug('Coupling graph is not a sequence of pairs.')
            return False

        if num_qudits is not None:
            if not (is_integer(num_qudits) and num_qudits > 0):
                _logger.debug('Invalid num_qudits in coupling graph check.')
                return False

            if not all(
                qudit < num_qudits
                for pair in coupling_graph
                for qudit in pair
            ):
                _logger.debug('Coupling graph has invalid qudits.')
                return False

        if not all([
            len(pair) == len(set(pair))
            for pair in coupling_graph
        ]):
            _logger.debug('Coupling graph has an invalid pair.')
            return False

        return True

    @staticmethod
    def all_to_all(num_qudits: int) -> CouplingGraph:
        """Return a coupling graph with all qudits connected."""
        return CouplingGraph(set(it.combinations(range(num_qudits), 2)))

    @staticmethod
    def linear(num_qudits: int) -> CouplingGraph:
        """Return a coupling graph with nearest-neighbor connectivity."""
        return CouplingGraph([(x, x + 1) for x in range(num_qudits - 1)])

    @staticmethod
    def ring(num_qudits: int) -> CouplingGraph:
        """Return a coupling graph with ring connectivity."""
        ext = [(0, num_qudits - 1)]
        return CouplingGraph([(x, x + 1) for x in range(num_qudits - 1)] + ext)

    @staticmethod
    def star(num_qudits: int) -> CouplingGraph:
        """Return a coupling graph with one qudit connected to all else."""
        return CouplingGraph([(0, x) for x in range(1, num_qudits)])

    @staticmethod
    def grid(num_rows: int, num_cols: int) -> CouplingGraph:
        """Return a coupling graph with a grid of qubits."""
        num_qudits = num_rows * num_cols
        edges = set()

        for i in range(num_qudits):
            if i % num_cols != num_cols - 1:
                edges.add((i, i + 1))

            if i < (num_rows - 1) * num_cols:
                edges.add((i, i + num_cols))

        return CouplingGraph(edges)

    def maximal_matching(
        self,
        edges_to_ignore: list[tuple[int, int]] = [],
        randomize: bool = False,
    ) -> list[tuple[int, int]]:
        """
        Generate a random graph matching for the coupling graph. Edges in the
        `ignored_edges` list will not be included.

        Arguments:
            edges_to_ignore (list[tuple[int]]): Edges not included in the
                matching. (Default: [])

            randomize (bool): Shuffle edges to create random matchings.
                (Default: False)

        Returns:
            matching (list[tuple[int]]): A maximal list of edges that share
                no common verticies.
        """
        matching: set[tuple[int, int]] = set()
        vertices: set[int] = set()

        edge_list = [
            (u, v) for (u, v) in self._edges if (u, v) not in edges_to_ignore
            and (v, u) not in edges_to_ignore
        ]
        if randomize:
            shuffle(edge_list)

        for edge in edge_list:
            u, v = edge
            if u not in vertices and v not in vertices and u != v:
                matching.add(edge)
                vertices.update(edge)
        return list(matching)

    def is_fully_connected_without(self, qudit: int) -> bool:
        """Return true if the graph is fully connected without `qudit`."""
        frontier = {0} if qudit != 0 else {1}
        qudits_seen = {0} if qudit != 0 else {1}

        while len(frontier) > 0:
            expanded_qudits = set()

            for q in frontier:
                if q != qudit:
                    neighbors = set(self.get_neighbors_of(q)) - {qudit}
                    expanded_qudits.update(neighbors)

            frontier = expanded_qudits - qudits_seen
            qudits_seen.update(expanded_qudits)

            if len(qudits_seen) == self.num_qudits - 1:
                return True

        return False

    def get_rooted_minimum_span(self, root: int) -> list[tuple[int, int]]:
        """
        Connect `root` to every other node in the graph.

        Args:
            root (int): The qudit to start from.

        Return:
            (list[tuple[int, int]]): A list of minimal edges that
                connects 'root' to every other qudit.
        """
        # Construct Minimum Spanning Tree using breadth-first search
        mst = []
        seen = {root}
        frontier = [root]

        while len(frontier) > 0 and len(seen) < self.num_qudits:
            qudit = frontier.pop(0)
            neighbors = set(self.get_neighbors_of(qudit))
            unseen_neighbors = neighbors - seen
            frontier.extend(unseen_neighbors)
            seen.update(neighbors)
            for neighbor in unseen_neighbors:
                mst.append((qudit, neighbor))

        mst = CouplingGraph(mst)

        # Traverse MST depth-first to produce pairs of interacting qudits
        interactions: list[tuple[int, int]] = []
        frontier: list[tuple[int, tuple[int, int] | None]] = [(root, None)]

        while len(frontier) > 0:
            qudit, interaction = frontier.pop(0)
            if interaction is not None:
                interactions.append(interaction)

            for neighbor in mst.get_neighbors_of(qudit):
                if interaction is None or neighbor != interaction[0]:
                    frontier.insert(0, (neighbor, (qudit, neighbor)))

        return interactions

    def get_induced_subgraph(
        self,
        location: CircuitLocationLike,
    ) -> list[tuple[int, int]]:
        """
        Return the edges induced by the vertices specified in `location`.

        Arguments:
            location (CircuitLocationLike): A list of qubits or vertices in the
                CouplingGraph. The edges connecting each of these vertices
                together will be extracted.

        Returns:
            (list[tuple[int,int]]): The list of edges connecting vertices in
                `location`.
        """
        warnings.warn(
        "get_induced_subgraph is deprecated and will be removed in a future release. "
        "Use get_subgraph(location) instead.",
        DeprecationWarning,
        stacklevel=2,
        )
        if not isinstance(location, CircuitLocation):
            location = CircuitLocation(location)

        if len(location) < 2:
            raise ValueError('Invalid location size.')

        subgraph = []
        for q1, q2 in it.combinations(location, 2):
            if (q1, q2) in self._edges or (q2, q1) in self._edges:
                subgraph.append((min([q1, q2]), max([q1, q2])))
        return subgraph

    def relabel_subgraph(
        graph: CouplingGraphLike,
        relabeling: dict[int, int] | None = None,
    ) -> CouplingGraph:
        """
        Renumber the vertices in `graph` according to the optionally provided
        `relabeling` dictionary, or relabel so that the vertices are in
        renumbered in least to greatest order and in the set {0,...,|V|-1}.

        Arguments:
            graph (CouplingGraphLike): A collection of edges representing a
                graph.

            relabeling (dict[int,int]|None): An optional dictionary specifying
                current labels as keys and desired labels as values.
                (Default: None)

        Returns:
            (CouplingGraph): The relabeled CouplingGraph.
        """
        if relabeling is None:
            vertices = set()
            for q1, q2 in graph:
                vertices.add(q1)
                vertices.add(q2)
            renum = {q: i for i, q in enumerate(vertices)}
        else:
            renum = relabeling

        edges = [
            (min([renum[q1], renum[q2]]), max([renum[q1], renum[q2]]))
            for q1, q2 in graph
        ]
        return CouplingGraph(edges)

    def is_embedded_in(self, graph: CouplingGraph) -> bool:
        """
        Check if this CouplingGraph is embedded within `graph`.

        Arguments:
            graph (CouplingGraph): A graph that may have a subgraph isomorphic
                to self.

        Returns:
            (bool): True if this graph is isomorphic to a subgraph of `graph`,
                and False if not.

        Note:
            The algorithm used is naive and relies on the fact that `graph` and
            `self` are similar in size and sparse.

        Todo:
            Implement the V2 algorithm for subgraph isomorphism.
        """
        if type(graph) is not CouplingGraph:
            raise ValueError('Provided `graph` is not of type CouplingGraph.')
        # A larger graph cannot be embedded in a smaller graph
        if self.num_qudits > graph.num_qudits:
            return False

        degs = {q: len(self._adj[q]) for q in range(self.num_qudits)}
        other_degs = {q: len(graph._adj[q]) for q in range(graph.num_qudits)}

        # Candidates are vertices in the other graph that have degree >= a
        # given vertex in self.
        candidate_labels: dict[int, list[int]] = {
            q: [] for q in range(self.num_qudits)
        }

        for q1 in range(self.num_qudits):
            for q2 in range(graph.num_qudits):
                if degs[q1] <= other_degs[q2]:
                    candidate_labels[q1].append(q2)

        # Each vertex must have at least one candidate vertex
        if any([len(cands) == 0 for cands in candidate_labels.values()]):
            return False

        # Check if the current renumbering works
        for renumbering in it.permutations(
            range(graph.num_qudits), self.num_qudits,
        ):
            renum = {q: renumbering[q] for q in range(self.num_qudits)}
            if all([
                (min([renum[u], renum[v]]), max([renum[u], renum[v]]))
                in graph for u, v in self._edges
            ]):
                return True
        return False


CouplingGraphLike = Union[Iterable[Tuple[int, int]], CouplingGraph]

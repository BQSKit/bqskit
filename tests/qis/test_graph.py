"""This module tests the CouplingGraph class."""
from __future__ import annotations

from typing import Any

import pytest

from bqskit.qis.graph import CouplingGraph
from bqskit.qis.graph import CouplingGraphLike


def test_coupling_graph_init_valid() -> None:
    # Test with valid inputs
    graph = {(0, 1), (1, 2), (2, 3)}
    num_qudits = 4
    remote_edges = [(1, 2)]
    default_weight = 1.0
    default_remote_weight = 10.0
    edge_weights_overrides = {(1, 2): 0.5}

    coupling_graph = CouplingGraph(
        graph,
        num_qudits,
        remote_edges,
        default_weight,
        default_remote_weight,
        edge_weights_overrides,
    )

    assert coupling_graph.num_qudits == num_qudits
    assert coupling_graph._edges == graph
    assert coupling_graph._remote_edges == set(remote_edges)
    assert coupling_graph.default_weight == default_weight
    assert coupling_graph.default_remote_weight == default_remote_weight
    assert all(
        coupling_graph._mat[q1][q2] == weight
        for (q1, q2), weight in edge_weights_overrides.items()
    )


@pytest.mark.parametrize(
    'graph, num_qudits, remote_edges, default_weight, default_remote_weight,'
    ' edge_weights_overrides, expected_exception',
    [
        # Invalid graph
        (None, 4, [], 1.0, 100.0, {}, TypeError),
        # num_qudits is not an integer
        ({(0, 1)}, '4', [], 1.0, 100.0, {}, TypeError),
        # num_qudits is negative
        ({(0, 1)}, -1, [], 1.0, 100.0, {}, ValueError),
        # Invalid remote_edges
        ({(0, 1)}, 4, None, 1.0, 100.0, {}, TypeError),
        # Remote edge not in graph
        ({(0, 1)}, 4, [(1, 2)], 1.0, 100.0, {}, ValueError),
        # Invalid default_weight
        ({(0, 1)}, 4, [], '1.0', 100.0, {}, TypeError),
        # Invalid default_remote_weight
        ({(0, 1)}, 4, [], 1.0, '100.0', {}, TypeError),
        # Invalid edge_weights_overrides
        ({(0, 1)}, 4, [], 1.0, 100.0, None, TypeError),
        # Non-integer value in edge_weights_overrides
        ({(0, 1)}, 4, [], 1.0, 100.0, {(0, 1): '0.5'}, TypeError),
        # Edge in edge_weights_overrides not in graph
        ({(0, 1)}, 4, [], 1.0, 100.0, {(1, 2): 0.5}, ValueError),
    ],
)
def test_coupling_graph_init_invalid(
    graph: CouplingGraphLike,
    num_qudits: Any,
    remote_edges: Any,
    default_weight: Any,
    default_remote_weight: Any,
    edge_weights_overrides: Any,
    expected_exception: Exception,
) -> None:
    with pytest.raises(expected_exception):
        CouplingGraph(
            graph,
            num_qudits,
            remote_edges,
            default_weight,
            default_remote_weight,
            edge_weights_overrides,
        )


def test_get_qpu_to_qudit_map_single_qpu() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)])
    expected_map = [[0, 1, 2, 3]]
    assert graph.get_qpu_to_qudit_map() == expected_map


def test_get_qpu_to_qudit_map_multiple_qpus() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    expected_map = [[0, 1], [2, 3]]
    assert graph.get_qpu_to_qudit_map() == expected_map


def test_get_qpu_to_qudit_map_disconnected() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (3, 4)], remote_edges=[(1, 2)])
    expected_map = [[0, 1], [2], [3, 4]]
    assert graph.get_qpu_to_qudit_map() == expected_map


def test_get_qpu_to_qudit_map_empty_graph() -> None:
    graph = CouplingGraph([])
    expected_map = [[0]]
    assert graph.get_qpu_to_qudit_map() == expected_map


def test_get_qpu_to_qudit_map_complex_topology() -> None:
    graph = CouplingGraph(
        [(0, 1), (1, 2), (0, 2), (2, 5), (3, 4), (4, 5), (3, 5)],
        remote_edges=[(2, 5)],
    )
    expected_map = [[0, 1, 2], [3, 4, 5]]
    assert graph.get_qpu_to_qudit_map() == expected_map


def test_get_qudit_to_qpu_map_three_qpu() -> None:
    graph = CouplingGraph(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        remote_edges=[(2, 3), (5, 6)],
    )
    expected_map = [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert graph.get_qpu_to_qudit_map() == expected_map


def test_is_distributed() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)])
    assert not graph.is_distributed()

    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    assert graph.is_distributed()

    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    assert graph.is_distributed()

    graph = CouplingGraph(
        [(0, 1), (1, 2), (2, 3)],
        remote_edges=[(1, 2), (2, 3)],
    )
    assert graph.is_distributed()


def test_qpu_count() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)])
    assert graph.qpu_count() == 1

    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    assert graph.qpu_count() == 2

    graph = CouplingGraph(
        [(0, 1), (1, 2), (2, 3)],
        remote_edges=[(1, 2), (2, 3)],
    )
    assert graph.qpu_count() == 3

    graph = CouplingGraph([])
    assert graph.qpu_count() == 1


def test_get_individual_qpu_graphs() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)])
    qpus = graph.get_individual_qpu_graphs()
    assert len(qpus) == 1
    assert qpus[0] == graph

    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    qpus = graph.get_individual_qpu_graphs()
    assert len(qpus) == 2
    assert qpus[0] == CouplingGraph([(0, 1)])
    assert qpus[1] == CouplingGraph([(0, 1)])

    graph = CouplingGraph(
        [(0, 1), (1, 2), (2, 3)],
        remote_edges=[(1, 2), (2, 3)],
    )
    qpus = graph.get_individual_qpu_graphs()
    assert len(qpus) == 3
    assert qpus[0] == CouplingGraph([(0, 1)])
    assert qpus[1] == CouplingGraph([])
    assert qpus[2] == CouplingGraph([])


def test_get_qudit_to_qpu_map() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)])
    assert graph.get_qudit_to_qpu_map() == [0, 0, 0, 0]

    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    assert graph.get_qudit_to_qpu_map() == [0, 0, 1, 1]

    graph = CouplingGraph(
        [(0, 1), (1, 2), (2, 3)],
        remote_edges=[(1, 2), (2, 3)],
    )
    assert graph.get_qudit_to_qpu_map() == [0, 0, 1, 2]

    graph = CouplingGraph([])
    assert graph.get_qudit_to_qpu_map() == [0]


def test_get_qpu_connectivity() -> None:
    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)])
    assert graph.get_qpu_connectivity() == [set()]

    graph = CouplingGraph([(0, 1), (1, 2), (2, 3)], remote_edges=[(1, 2)])
    assert graph.get_qpu_connectivity() == [{1}, {0}]

    graph = CouplingGraph(
        [(0, 1), (1, 2), (2, 3)],
        remote_edges=[(1, 2), (2, 3)],
    )
    assert graph.get_qpu_connectivity() == [{1}, {0, 2}, {1}]


class TestGraphGetSubgraphsOfSize:
    def test_1(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})
        l = coupling_graph.get_subgraphs_of_size(2)

        assert len(l) == 3
        assert (0, 1) in l
        assert (1, 2) in l
        assert (2, 3) in l

    def test_2(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})
        l = coupling_graph.get_subgraphs_of_size(3)

        assert len(l) == 2
        assert (0, 1, 2) in l
        assert (1, 2, 3) in l

    def test_3(self) -> None:
        coupling_graph = CouplingGraph.all_to_all(4)
        l = coupling_graph.get_subgraphs_of_size(3)

        assert len(l) == 4
        assert (0, 1, 2) in l
        assert (0, 1, 3) in l
        assert (0, 2, 3) in l
        assert (1, 2, 3) in l

    def test_invalid(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})

        with pytest.raises(TypeError):
            coupling_graph.get_subgraphs_of_size('a')  # type: ignore


class TestMachineGetSubgraph:
    def test_1(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})
        l = coupling_graph.get_subgraph((0, 1, 2))

        assert len(l) == 2
        assert (0, 1) in l
        assert (1, 2) in l

    def test_2(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (0, 3), (2, 3)})
        l = coupling_graph.get_subgraph((0, 1, 3))

        assert len(l) == 2
        assert (0, 1) in l
        assert (0, 2) in l

    def test_3(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (0, 3), (2, 3)})
        renumbering = {0: 1, 1: 2, 3: 0}
        l = coupling_graph.get_subgraph((0, 1, 3), renumbering)

        assert len(l) == 2
        assert (0, 1) in l
        assert (1, 2) in l

    def test_invalid_1(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})

        with pytest.raises(TypeError):
            coupling_graph.get_subgraph('a')  # type: ignore

    def test_invalid_2(self) -> None:
        coupling_graph = CouplingGraph.all_to_all(5)

        with pytest.raises(ValueError):
            renumbering = {1: 1, 3: 3, 4: 4}
            coupling_graph.get_subgraph((1, 3, 4), renumbering)

    def test_invalid_3(self) -> None:
        coupling_graph = CouplingGraph.all_to_all(5)

        with pytest.raises(ValueError):
            renumbering = {0: 0, 1: 1, 2: 2}
            coupling_graph.get_subgraph((1, 3, 4), renumbering)

    def test_invalid_4(self) -> None:
        coupling_graph = CouplingGraph.all_to_all(5)

        with pytest.raises(ValueError):
            renumbering = {0: 0, 1: 1, 3: 3, 4: 4}
            coupling_graph.get_subgraph((1, 3, 4), renumbering)


def test_is_linear() -> None:
    coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})
    assert coupling_graph.is_linear()

    coupling_graph = CouplingGraph({(0, 1), (1, 2), (0, 3), (2, 3)})
    assert not coupling_graph.is_linear()

    coupling_graph = CouplingGraph.all_to_all(4)
    assert not coupling_graph.is_linear()

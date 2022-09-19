"""This module tests the CouplingGraph class."""
from __future__ import annotations

import pytest

from bqskit.qis.graph import CouplingGraph


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

    def test_invalid(self) -> None:
        coupling_graph = CouplingGraph({(0, 1), (1, 2), (2, 3)})

        with pytest.raises(TypeError):
            coupling_graph.get_subgraph('a')  # type: ignore

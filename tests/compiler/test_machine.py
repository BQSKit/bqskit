from __future__ import annotations

import pytest

from bqskit.compiler.machine import MachineModel


class TestMachineConstructor:
    def test_coupling_graph(self) -> None:
        coupling_graph = {(0, 1), (1, 2), (2, 3)}
        t = MachineModel(4, coupling_graph)

        assert len(t.coupling_graph) == 3
        for link in coupling_graph:
            assert link in t.coupling_graph

    def test_num_qudits(self) -> None:
        for n in [1, 2, 3, 4]:
            t = MachineModel(n)
            assert t.num_qudits == n

    def test_alltoall_2(self) -> None:
        t = MachineModel(2)
        assert len(t.coupling_graph) == 1
        assert (0, 1) in t.coupling_graph

    def test_alltoall_3(self) -> None:
        t = MachineModel(3)
        assert len(t.coupling_graph) == 3
        assert (0, 1) in t.coupling_graph
        assert (0, 2) in t.coupling_graph
        assert (1, 2) in t.coupling_graph

    def test_alltoall_4(self) -> None:
        t = MachineModel(4)
        assert len(t.coupling_graph) == 6
        assert (0, 1) in t.coupling_graph
        assert (0, 2) in t.coupling_graph
        assert (0, 3) in t.coupling_graph
        assert (1, 2) in t.coupling_graph
        assert (1, 3) in t.coupling_graph
        assert (2, 3) in t.coupling_graph

    def test_num_qudits_invalid(self) -> None:
        with pytest.raises(TypeError):
            MachineModel('a')  # type: ignore
        with pytest.raises(ValueError):
            MachineModel(0)

    def test_coupling_graph_invalid(self) -> None:
        coupling_graph = {(0, 1), (1, 2), (2, 3)}
        with pytest.raises(TypeError):
            MachineModel(2, coupling_graph)
        with pytest.raises(TypeError):
            MachineModel(2, (0, 1))  # type: ignore
        with pytest.raises(TypeError):
            MachineModel(2, 0)  # type: ignore
        with pytest.raises(TypeError):
            MachineModel(2, 'a')  # type: ignore

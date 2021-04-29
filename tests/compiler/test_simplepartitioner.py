from __future__ import annotations

import pytest
import itertools as it

from bqskit.compiler.passes.simplepartitioner import SimplePartitioner
from bqskit.compiler.machine import MachineModel

class TestMachineConstructor:

    def test_constructor(self) -> None:
        mach = MachineModel(3)
        part = SimplePartitioner(mach, 3)
        assert part.b_size == 3

    def test_get_qudit_groups(self) -> None:
        num_q = 9
        b_size = 3
        mach = MachineModel(num_q)
        part = SimplePartitioner(mach, b_size)
        groups = part.get_qudit_groups()
        for group in groups:
            perms = it.combinations(group, 2)
            for perm in perms:
                assert perm in part.machine.coupling_graph \
                    or (perm[1], perm[0]) in part.machine.coupling_graph

    def test_get_used_qudit_set(self) -> None:
        pass

    def test_qudit_group_search(self) -> None:
        pass

    def test_set_run_parameters(self) -> None:
        pass

    def test_num_ops_left(self) -> None:
        pass

    def test_run(self) -> None:
        pass
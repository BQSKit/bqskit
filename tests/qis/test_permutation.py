from __future__ import annotations

from bqskit.ir.gates import SwapGate
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.unitary import UnitaryMatrix


def test_permutation_1() -> None:
    swap = SwapGate().get_unitary()
    id = UnitaryMatrix.identity(2)
    correct = id.otimes(swap)
    perm = PermutationMatrix.from_qubit_location(3, (0, 2, 1))
    assert perm == correct


def test_permutation_2() -> None:
    swap = SwapGate().get_unitary()
    correct = swap.otimes(swap)
    perm = PermutationMatrix.from_qubit_location(4, (1, 0, 3, 2))
    assert perm == correct


def test_permutation_3() -> None:
    swap = SwapGate().get_unitary()
    id = UnitaryMatrix.identity(2)
    correct = id.otimes(swap, id) @ swap.otimes(swap)
    perm = PermutationMatrix.from_qubit_location(4, (1, 3, 0, 2))
    assert perm == correct

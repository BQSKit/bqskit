"""This module tests the PermutationGate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given

from bqskit.ir.gates import PermutationGate
from bqskit.qis.permutation import PermutationMatrix
from bqskit.utils.test.strategies import num_qudits


@given(num_qudits())
def test_permutation(num_qudits: int) -> None:
    x = np.arange(num_qudits)
    np.random.shuffle(x)
    p = PermutationGate(num_qudits, tuple(x))
    assert p.num_qudits == num_qudits
    assert p.num_params == 0
    assert PermutationMatrix.is_permutation(p.get_unitary())

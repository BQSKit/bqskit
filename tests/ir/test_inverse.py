from __future__ import annotations

import numpy as np
from hypothesis import given

from bqskit.ir.operation import Operation
from bqskit.utils.test.strategies import operations


@given(operations())
def test_gate_inverse(op: Operation) -> None:
    inverse_op = op.get_inverse()
    iden = np.identity(op.dim)
    dist = (inverse_op.get_unitary() @ op.get_unitary()).get_distance_from(iden)
    assert dist < 1e-7

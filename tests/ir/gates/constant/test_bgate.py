"""This module tests the IdentityGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constant.b import BGate
from bqskit.utils.test.strategies import num_qudits_and_radixes


def test_identity() -> None:
    B = BGate()
    U = np.array([
        [0.92387953+0.j, 0.+0.j, 0.+0.j, 0.+0.38268343j]
        [0.+0.j, 0.38268343+0.j, 0.+0.92387953j, 0.+0.j]
        [0.+0.j, 0.+0.92387953j, 0.38268343+0.j, 0.+0.j]
        [0.+0.38268343j, 0.+0.j, 0.+0.j, 0.92387953+0.j]
    ])
    assert B.num_qudits == 2
    assert B.num_params == 0
    assert B.get_unitary() == U

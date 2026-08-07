"""This module tests the RYGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import RYGate
from bqskit.ir.gates import YGate


def test_get_unitary() -> None:
    g = RYGate()
    u = YGate().get_unitary()
    assert g.get_unitary([np.pi]).get_distance_from(u) < 1e-7

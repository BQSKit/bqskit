"""This module tests the RYYGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import RYYGate
from bqskit.ir.gates import YYGate


def test_get_unitary() -> None:
    g = RYYGate()
    u = YYGate().get_unitary()
    assert g.get_unitary([np.pi / 2]).get_distance_from(u) < 1e-7

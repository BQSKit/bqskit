"""This module tests the RXXGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import RXXGate
from bqskit.ir.gates import XXGate


def test_get_unitary() -> None:
    g = RXXGate()
    u = XXGate().get_unitary()
    assert g.get_unitary([np.pi / 2]).get_distance_from(u) < 1e-7

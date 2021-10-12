"""This module tests the RZZGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import RZZGate
from bqskit.ir.gates import ZZGate


def test_get_unitary() -> None:
    g = RZZGate()
    u = ZZGate().get_unitary()
    assert g.get_unitary([np.pi / 2]).get_distance_from(u) < 1e-7

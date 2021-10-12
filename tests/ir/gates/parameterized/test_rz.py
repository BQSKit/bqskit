"""This module tests the RZGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import RZGate
from bqskit.ir.gates import ZGate


def test_get_unitary() -> None:
    g = RZGate()
    u = ZGate().get_unitary()
    assert g.get_unitary([np.pi]).get_distance_from(u) < 1e-7

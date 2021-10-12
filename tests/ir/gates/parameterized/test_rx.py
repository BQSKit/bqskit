"""This module tests the RXGate class."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates import RXGate
from bqskit.ir.gates import XGate


def test_get_unitary() -> None:
    g = RXGate()
    u = XGate().get_unitary()
    assert g.get_unitary([np.pi]).get_distance_from(u) < 1e-7

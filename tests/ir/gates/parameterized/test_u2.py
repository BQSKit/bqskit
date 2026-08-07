"""This module tests the U2Gate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U2Gate


@given(
    floats(allow_nan=False, allow_infinity=False, width=32),
    floats(allow_nan=False, allow_infinity=False, width=32),
)
def test_get_unitary(angle1: float, angle2: float) -> None:
    u = U2Gate().get_unitary([angle1, angle2])
    z1 = RZGate().get_unitary([angle1])
    y = RYGate().get_unitary([np.pi / 2])
    z2 = RZGate().get_unitary([angle2])
    assert u.get_distance_from(z1 @ y @ z2) < 1e-7

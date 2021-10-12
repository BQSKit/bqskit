"""This module tests the U1Gate class."""
from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U1Gate


@given(floats(allow_nan=False, allow_infinity=False, width=32))
def test_get_unitary(angle: float) -> None:
    u = U1Gate()
    z = RZGate()
    dist = u.get_unitary([angle]).get_distance_from(z.get_unitary([angle]))
    assert dist < 1e-7

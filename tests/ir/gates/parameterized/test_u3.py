"""This module tests the U3Gate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import U3Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.test.strategies import unitaries


@given(
    floats(allow_nan=False, allow_infinity=False, width=32),
    floats(allow_nan=False, allow_infinity=False, width=32),
    floats(allow_nan=False, allow_infinity=False, width=32),
)
def test_get_unitary(angle1: float, angle2: float, angle3: float) -> None:
    u = U3Gate().get_unitary([angle1, angle2, angle3])
    z1 = RZGate().get_unitary([angle1])
    x1 = RXGate().get_unitary([-np.pi / 2])
    z2 = RZGate().get_unitary([angle2])
    x2 = RXGate().get_unitary([np.pi / 2])
    z3 = RZGate().get_unitary([angle3])
    assert u.get_distance_from(z2 @ x1 @ z1 @ x2 @ z3) < 1e-7


@given(unitaries(1, (2,)))
def test_calc_params(utry: UnitaryMatrix) -> None:
    params = U3Gate.calc_params(utry)
    assert U3Gate().get_unitary(params).get_distance_from(utry) < 1e-7

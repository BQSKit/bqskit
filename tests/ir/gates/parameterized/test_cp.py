"""This module tests the U1Gate class."""
from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import CPGate
from bqskit.qis.unitary import UnitaryMatrix
import numpy as np


@given(floats(allow_nan=False, allow_infinity=False, width=32))
def test_get_unitary(angle: float) -> None:
    utry = UnitaryMatrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * angle)],
        ]
    )
    cp = CPGate()
    assert cp.num_params == 1
    assert cp.num_qudits == 2
    dist = cp.get_unitary([angle]).get_distance_from(utry)
    assert dist < 1e-7

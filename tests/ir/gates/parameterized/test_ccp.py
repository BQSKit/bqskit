"""This module tests the U1Gate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import CCPGate
from bqskit.qis.unitary import UnitaryMatrix


@given(floats(allow_nan=False, allow_infinity=False, width=32))
def test_get_unitary(angle: float) -> None:
    utry = UnitaryMatrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, np.exp(1j * angle)],
        ],
    )
    ccp = CCPGate()
    assert ccp.num_params == 1
    assert ccp.num_qudits == 3
    dist = ccp.get_unitary([angle]).get_distance_from(utry)
    assert dist < 1e-7

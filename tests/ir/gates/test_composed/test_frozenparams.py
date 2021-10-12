"""This module tests the FrozenParameterGate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import FrozenParameterGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import U3Gate


@given(floats(allow_nan=False, allow_infinity=False))
def test_u3_rx(angle: float) -> None:
    frozen_u3 = FrozenParameterGate(U3Gate(), {1: -np.pi / 2, 2: np.pi / 2})
    assert frozen_u3.get_unitary([angle]) == RXGate().get_unitary([angle])
    assert frozen_u3.num_params == 1

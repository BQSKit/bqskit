"""This module tests the FrozenParameterGate class."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats

from bqskit.ir.gates import FrozenParameterGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import U3Gate


@given(
    floats(
        min_value=-1e15,
        max_value=1e15,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_u3_rx(angle: float) -> None:
    # Bounded away from float extremes: near f64::MAX, RXGate's openqudit
    # expression correctly raises ValueError rather than returning a
    # matrix that fails the unitarity check (floating-point precision is
    # fundamentally insufficient at that magnitude), whereas U3Gate's
    # hand-written numpy path has no such check. No real rotation angle
    # is anywhere near this range.
    frozen_u3 = FrozenParameterGate(U3Gate(), {1: -np.pi / 2, 2: np.pi / 2})
    assert frozen_u3.get_unitary([angle]) == RXGate().get_unitary([angle])
    assert frozen_u3.num_params == 1

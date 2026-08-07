"""This module tests the U3Gate class."""
from __future__ import annotations

from bqskit.ir.gates import U8Gate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def test_calc_params() -> None:
    for _ in range(100):
        U = UnitaryMatrix.random(1, [3])
        params = U8Gate().calc_params(U)
        assert U8Gate().get_unitary(params).get_distance_from(U) < 1e-7

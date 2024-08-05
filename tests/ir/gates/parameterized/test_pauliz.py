"""This module tests the PauliZGate class."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates import PauliZGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import RZZGate
from bqskit.utils.test.strategies import num_qudits


class TestInit:
    @given(num_qudits(4))
    def test_valid(self, num_qudits: int) -> None:
        g = PauliZGate(num_qudits)
        assert g.num_qudits == num_qudits
        assert g.num_params == 2 ** num_qudits
        identity = np.identity(2 ** num_qudits)
        assert g.get_unitary([0] * 2 ** num_qudits) == identity

    @given(integers(max_value=0))
    def test_invalid(self, num_qudits: int) -> None:
        with pytest.raises(ValueError):
            PauliZGate(num_qudits)


class TestGetUnitary:
    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_i(self, angle: float) -> None:
        g = PauliZGate(1)
        i = IdentityGate(1).get_unitary()
        dist = g.get_unitary([angle, 0]).get_distance_from(i)
        assert dist < 1e-7

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_z(self, angle: float) -> None:
        g = PauliZGate(1)
        z = RZGate()
        assert g.get_unitary([0, angle]) == z.get_unitary([angle])

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_zz(self, angle: float) -> None:
        g = PauliZGate(2)
        zz = RZZGate()
        params = [0.0] * 4
        params[3] = angle
        assert g.get_unitary(params) == zz.get_unitary([angle])


@given(floats(allow_nan=False, allow_infinity=False, width=16))
def test_optimize(angle: float) -> None:
    g = PauliZGate(1)
    z = RZGate()
    utry = z.get_unitary([angle])
    params = g.optimize(np.array(utry))
    assert g.get_unitary(params).get_distance_from(utry.conj().T) < 1e-7

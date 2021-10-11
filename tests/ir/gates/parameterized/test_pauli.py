"""This module tests the PauliGate class."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates import PauliGate
from bqskit.ir.gates import RXGate
from bqskit.ir.gates import RXXGate
from bqskit.ir.gates import RYGate
from bqskit.ir.gates import RYYGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import RZZGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.test.strategies import num_qudits
from bqskit.test.strategies import unitaries


class TestInit:
    @given(num_qudits(4))
    def test_valid(self, num_qudits: int) -> None:
        g = PauliGate(num_qudits)
        assert g.num_qudits == num_qudits
        assert g.num_params == 4 ** num_qudits
        identity = np.identity(2 ** num_qudits)
        assert g.get_unitary([0] * 4 ** num_qudits) == identity

    @given(integers(max_value=0))
    def test_invalid(self, num_qudits: int) -> None:
        with pytest.raises(ValueError):
            PauliGate(num_qudits)


class TestGetUnitary:
    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_i(self, angle: float) -> None:
        g = PauliGate(1)
        i = IdentityGate(1).get_unitary()
        dist = g.get_unitary([angle, 0, 0, 0]).get_distance_from(i)
        assert dist < 1e-7

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_x(self, angle: float) -> None:
        g = PauliGate(1)
        x = RXGate()
        assert g.get_unitary([0, angle, 0, 0]) == x.get_unitary([angle])

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_y(self, angle: float) -> None:
        g = PauliGate(1)
        y = RYGate()
        assert g.get_unitary([0, 0, angle, 0]) == y.get_unitary([angle])

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_z(self, angle: float) -> None:
        g = PauliGate(1)
        z = RZGate()
        assert g.get_unitary([0, 0, 0, angle]) == z.get_unitary([angle])

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_xx(self, angle: float) -> None:
        g = PauliGate(2)
        xx = RXXGate()
        params = [0.0] * 16
        params[5] = angle
        assert g.get_unitary(params) == xx.get_unitary([angle])

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_yy(self, angle: float) -> None:
        g = PauliGate(2)
        yy = RYYGate()
        params = [0.0] * 16
        params[10] = angle
        assert g.get_unitary(params) == yy.get_unitary([angle])

    @given(floats(allow_nan=False, allow_infinity=False, width=16))
    def test_zz(self, angle: float) -> None:
        g = PauliGate(2)
        zz = RZZGate()
        params = [0.0] * 16
        params[15] = angle
        assert g.get_unitary(params) == zz.get_unitary([angle])


@given(unitaries(1, (2,)))
def test_optimize(utry: UnitaryMatrix) -> None:
    g = PauliGate(1)
    params = g.optimize(np.array(utry))
    assert g.get_unitary(params).get_distance_from(utry.conj().T) < 1e-7

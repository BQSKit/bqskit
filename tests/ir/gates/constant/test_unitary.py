"""This module tests the ConstantUnitaryGate class."""
from __future__ import annotations

from hypothesis import given

from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.test.strategies import unitaries


@given(unitaries())
def test_constant_unitary(utry: UnitaryMatrix) -> None:
    u = ConstantUnitaryGate(utry)
    assert u.num_qudits == utry.num_qudits
    assert u.radixes == utry.radixes
    assert u.num_params == 0
    assert u.get_unitary() == u


@given(unitaries())
def test_constant_unitary_like(utry: UnitaryMatrix) -> None:
    u = ConstantUnitaryGate(utry.numpy, utry.radixes)
    assert u.num_qudits == utry.num_qudits
    assert u.radixes == utry.radixes
    assert u.num_params == 0
    assert u.get_unitary() == u

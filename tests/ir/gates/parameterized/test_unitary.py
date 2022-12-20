"""This module tests the VariableUnitaryGate class."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.test.strategies import num_qudits_and_radixes
from bqskit.utils.test.strategies import unitaries


class TestInit:
    @given(num_qudits_and_radixes(3, (2, 3)))
    def test_valid(self, pair: tuple[int, tuple[int, ...]]) -> None:
        num_qudits, radixes = pair
        vug = VariableUnitaryGate(num_qudits, radixes)
        assert vug.num_qudits == num_qudits
        assert vug.radixes == radixes
        assert vug.num_params == 2 * int(np.prod(radixes))**2

    @given(integers(max_value=0))
    def test_invalid(self, num_qudits: int) -> None:
        with pytest.raises(ValueError):
            VariableUnitaryGate(num_qudits)


class TestGetUnitary:
    @given(unitaries())
    def test_exact(self, utry: UnitaryMatrix) -> None:
        params = list(np.reshape(np.real(utry.numpy), (-1,))) + \
            list(np.reshape(np.imag(utry.numpy), (-1,)))
        vug = VariableUnitaryGate(utry.num_qudits, utry.radixes)
        assert vug.get_unitary(params) == utry

    @given(unitaries())
    def test_phase(self, utry: UnitaryMatrix) -> None:
        utry2 = -1 * utry
        params = list(np.reshape(np.real(utry2.numpy), (-1,))) + \
            list(np.reshape(np.imag(utry2.numpy), (-1,)))
        vug = VariableUnitaryGate(utry.num_qudits, utry.radixes)
        assert vug.get_unitary(params).get_distance_from(utry) < 1e-7


@given(unitaries())
def test_optimize(utry: UnitaryMatrix) -> None:
    vug = VariableUnitaryGate(utry.num_qudits, utry.radixes)
    params = vug.optimize(np.array(utry))
    assert vug.get_unitary(params).get_distance_from(utry.conj().T) < 1e-7


@given(unitaries(2, (2,), 2))
def test_is(utry: UnitaryMatrix) -> None:
    vug1 = VariableUnitaryGate(utry.num_qudits, utry.radixes)
    vug2 = VariableUnitaryGate(utry.num_qudits)
    assert vug1 == vug2

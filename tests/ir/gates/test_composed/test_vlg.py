"""This module tests the VariableLocationGate class."""
from __future__ import annotations

from hypothesis import given

from bqskit.ir.gates import VariableLocationGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.test.strategies import unitaries


@given(unitaries(2, (2,)).filter(lambda x: x.num_qudits == 2))
def test_vlg(utry: UnitaryMatrix) -> None:
    vlg = VariableLocationGate(ConstantUnitaryGate(utry), [(0, 1), (1, 2)])
    assert vlg.num_params == 2
    assert vlg.num_qudits == 3
    UI = utry.otimes(UnitaryMatrix.identity(2))
    IU = UnitaryMatrix.identity(2).otimes(utry)
    assert vlg.get_unitary([100, 0]).get_distance_from(UI) < 1e-7
    assert vlg.get_unitary([0, 100]).get_distance_from(IU) < 1e-7

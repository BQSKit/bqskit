"""This module tests the TaggedGate class."""
from __future__ import annotations

from hypothesis import given

from bqskit.ir.gates import TaggedGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.test.strategies import unitaries


@given(unitaries())
def test_tag(utry: UnitaryMatrix) -> None:
    tagged_gate = TaggedGate(ConstantUnitaryGate(utry), 'test')
    assert tagged_gate.tag == 'test'
    assert tagged_gate.get_unitary() == utry

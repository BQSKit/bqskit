from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import integers

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicates.width import WidthPredicate


@given(integers(0, 32), integers(1, 32))
def test_width_predicate(x: int, y: int) -> None:
    pred = WidthPredicate(x)
    assert pred.get_truth_value(Circuit(y), PassData(Circuit(y))) == (y < x)

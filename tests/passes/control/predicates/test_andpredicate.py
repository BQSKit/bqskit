from __future__ import annotations

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.control.predicates import AndPredicate


class TruePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return True


class FalsePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return False


def test_true_and_true() -> None:
    circuit = Circuit(1)
    data = PassData(circuit)
    pred = AndPredicate(TruePredicate(), TruePredicate())
    assert pred.get_truth_value(circuit, data)


def test_true_and_false() -> None:
    circuit = Circuit(1)
    data = PassData(circuit)
    pred = AndPredicate(TruePredicate(), FalsePredicate())
    assert not pred.get_truth_value(circuit, data)


def test_false_and_true() -> None:
    circuit = Circuit(1)
    data = PassData(circuit)
    pred = AndPredicate(FalsePredicate(), TruePredicate())
    assert not pred.get_truth_value(circuit, data)


def test_false_and_false() -> None:
    circuit = Circuit(1)
    data = PassData(circuit)
    pred = AndPredicate(FalsePredicate(), FalsePredicate())
    assert not pred.get_truth_value(circuit, data)

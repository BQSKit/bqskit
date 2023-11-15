from __future__ import annotations

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.control.predicates import NotPredicate


class TruePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return True


class FalsePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return False


def test_not_true() -> None:
    circuit = Circuit(1)
    data = PassData(circuit)
    assert not NotPredicate(TruePredicate()).get_truth_value(circuit, data)


def test_not_false() -> None:
    circuit = Circuit(1)
    data = PassData(circuit)
    assert NotPredicate(FalsePredicate()).get_truth_value(circuit, data)

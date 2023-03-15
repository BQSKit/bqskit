from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.passes.control.ifthenelse import IfThenElsePass
from bqskit.passes.control.predicate import PassPredicate


class FalsePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return False


class TruePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return True


class AddCXGate(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        circuit.append_gate(CXGate(), (0, 1))


class NeverRuns(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        assert False, 'Should never have been executed.'


def test_ifthenelse_on_true(compiler: Compiler) -> None:
    ite_pass = IfThenElsePass(TruePredicate(), AddCXGate())
    out_circuit = compiler.compile(Circuit(2), ite_pass)
    assert CXGate() in out_circuit.gate_set


def test_ifthenelse_on_true_with_false(compiler: Compiler) -> None:
    ite_pass = IfThenElsePass(TruePredicate(), AddCXGate(), NeverRuns())
    out_circuit = compiler.compile(Circuit(2), ite_pass)
    assert CXGate() in out_circuit.gate_set


def test_ifthenelse_on_false(compiler: Compiler) -> None:
    ite_pass = IfThenElsePass(FalsePredicate(), NeverRuns(), AddCXGate())
    out_circuit = compiler.compile(Circuit(2), ite_pass)
    assert CXGate() in out_circuit.gate_set

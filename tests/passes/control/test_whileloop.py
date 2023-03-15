from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.control.whileloop import WhileLoopPass


class FalsePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return False


class TrueX3Predicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        if '__true_x3_counter' not in data:
            data['__true_x3_counter'] = 1
            return True

        data['__true_x3_counter'] += 1
        return data['__true_x3_counter'] <= 3


class NeverRuns(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        assert False, 'Should never have been executed.'


class RunsThrice(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        if '_runs_thrice_counter' not in data:
            data['_runs_thrice_counter'] = 1
            return

        data['_runs_thrice_counter'] += 1
        if data['_runs_thrice_counter'] > 3:
            assert False, 'Should never have been executed.'


def test_while_never_runs_on_false(compiler: Compiler) -> None:
    wl_pass = WhileLoopPass(FalsePredicate(), NeverRuns())
    compiler.compile(Circuit(1), wl_pass)


def test_while_runs_while_true(compiler: Compiler) -> None:
    wl_pass = WhileLoopPass(TrueX3Predicate(), RunsThrice())
    compiler.compile(Circuit(1), wl_pass)

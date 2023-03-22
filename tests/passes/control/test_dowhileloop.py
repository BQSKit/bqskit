from __future__ import annotations

import pytest

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.control.dowhileloop import DoWhileLoopPass
from bqskit.passes.control.predicate import PassPredicate


class FalsePredicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        return False


class TrueX2Predicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        if '__true_x2_counter' not in data:
            data['__true_x2_counter'] = 1
            return True

        data['__true_x2_counter'] += 1
        return data['__true_x2_counter'] <= 2


class TrueX3Predicate(PassPredicate):
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        if '__true_x3_counter' not in data:
            data['__true_x3_counter'] = 1
            return True

        data['__true_x3_counter'] += 1
        return data['__true_x3_counter'] <= 3


class RunsOnce(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        if '_runs_once_flag' in data:
            assert False, 'Should never have been executed.'

        data['_runs_once_flag'] = True


class RunsThrice(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        if '_runs_thrice_counter' not in data:
            data['_runs_thrice_counter'] = 1
            return

        data['_runs_thrice_counter'] += 1
        if data['_runs_thrice_counter'] > 3:
            assert False, 'Should never have been executed.'


def test_dowhile_runs_once_on_false(compiler: Compiler) -> None:
    dwl_pass = DoWhileLoopPass(FalsePredicate(), RunsOnce())
    compiler.compile(Circuit(1), [dwl_pass])


def test_dowhile_runs_while_true(compiler: Compiler) -> None:
    dwl_pass = DoWhileLoopPass(TrueX2Predicate(), RunsThrice())
    compiler.compile(Circuit(1), [dwl_pass])


def test_dowhile_runs_while_true_xfail(fresh_compiler: Compiler) -> None:
    dwl_pass = DoWhileLoopPass(TrueX3Predicate(), RunsThrice())
    with pytest.raises(RuntimeError):
        fresh_compiler.compile(Circuit(1), [dwl_pass])

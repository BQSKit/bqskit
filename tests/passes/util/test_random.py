from __future__ import annotations

import pytest

from bqskit.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir import Circuit
from bqskit.ir.gates import CCXGate
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import SetRandomSeedPass
from bqskit.qis import UnitaryMatrix


def test_two_qubit_syn_with_seed(compiler: Compiler) -> None:
    in_utry = UnitaryMatrix.random(2)

    circ1 = Circuit.from_unitary(in_utry)
    task1 = CompilationTask(
        circ1, [SetRandomSeedPass(0), QSearchSynthesisPass()],
    )

    circ2 = Circuit.from_unitary(in_utry)
    task2 = CompilationTask(
        circ2, [SetRandomSeedPass(0), QSearchSynthesisPass()],
    )

    circ1 = compiler.compile(task1)
    circ2 = compiler.compile(task2)

    for op1, op2 in zip(circ1, circ2):
        assert op1 == op2


def test_three_qubit_syn_with_seed(compiler: Compiler) -> None:
    in_utry = CCXGate().get_unitary()

    circ1 = Circuit.from_unitary(in_utry)
    task1 = CompilationTask(
        circ1, [SetRandomSeedPass(0), QSearchSynthesisPass()],
    )

    circ2 = Circuit.from_unitary(in_utry)
    task2 = CompilationTask(
        circ2, [SetRandomSeedPass(0), QSearchSynthesisPass()],
    )

    circ1 = compiler.compile(task1)
    circ2 = compiler.compile(task2)

    for op1, op2 in zip(circ1, circ2):
        assert op1 == op2

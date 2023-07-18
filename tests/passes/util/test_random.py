from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CCXGate
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import SetRandomSeedPass
from bqskit.qis import UnitaryMatrix


def test_two_qubit_syn_with_seed(compiler: Compiler) -> None:
    in_utry = UnitaryMatrix.random(2)

    circ1 = Circuit.from_unitary(in_utry)
    circ2 = Circuit.from_unitary(in_utry)
    workflow = [SetRandomSeedPass(0), QSearchSynthesisPass()]
    circ1 = compiler.compile(circ1, workflow)
    circ2 = compiler.compile(circ2, workflow)

    for op1, op2 in zip(circ1, circ2):
        assert op1 == op2


def test_three_qubit_syn_with_seed(compiler: Compiler) -> None:
    in_utry = CCXGate().get_unitary()
    circ1 = Circuit.from_unitary(in_utry)
    circ2 = Circuit.from_unitary(in_utry)
    workflow = [SetRandomSeedPass(0), QSearchSynthesisPass()]
    circ1 = compiler.compile(circ1, workflow)
    circ2 = compiler.compile(circ2, workflow)

    for op1, op2 in zip(circ1, circ2):
        assert op1 == op2

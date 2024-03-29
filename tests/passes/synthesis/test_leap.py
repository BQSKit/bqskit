from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cpi import CPIGate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis import UnitaryMatrix


class TestLeap:

    def test_small_qubit(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(utry)
        leap = LEAPSynthesisPass()
        circuit = compiler.compile(circuit, [leap])
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_small_qutrit(self) -> None:
        utry = UnitaryMatrix.random(2, [3, 3])
        circuit = Circuit.from_unitary(utry)
        layer_gen = SimpleLayerGenerator(CPIGate(), U8Gate())
        leap = LEAPSynthesisPass(layer_generator=layer_gen)
        circuit.perform(leap)
        dist = circuit.get_unitary().get_distance_from(utry, 1)
        assert dist <= leap.success_threshold
